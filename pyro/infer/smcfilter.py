# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import contextlib
import math

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer.util import is_validation_enabled
from pyro.poutine.util import prune_subsample_sites


class SMCFailed(ValueError):
    """
    Exception raised when :class:`SMCFilter` fails to find any hypothesis with
    nonzero probability.
    """
    pass


class SMCFilter:
    """
    :class:`SMCFilter` is the top-level interface for filtering via sequential
    monte carlo.

    The model and guide should be objects with two methods:
    ``.init(state, ...)`` and ``.step(state, ...)``, intended to be called
    first with :meth:`init` , then with :meth:`step` repeatedly.  These two
    methods should have the same signature as :class:`SMCFilter` 's
    :meth:`init` and :meth:`step` of this class, but with an extra first
    argument ``state`` that should be used to store all tensors that depend on
    sampled variables. The ``state`` will be a dict-like object,
    :class:`SMCState` , with arbitrary keys and :class:`torch.Tensor` values.
    Models can read and write ``state`` but guides can only read from it.

    Inference complexity is ``O(len(state) * num_time_steps)``, so to avoid
    quadratic complexity in Markov models, ensure that ``state`` has fixed size.

    :param object model: probabilistic model with ``init`` and ``step`` methods
    :param object guide: guide used for sampling,  with ``init`` and ``step``
        methods
    :param int num_particles: The number of particles used to form the
        distribution.
    :param int max_plate_nesting: Bound on max number of nested
        :func:`pyro.plate` contexts.
    :param float ess_threshold: Effective sample size threshold for deciding
        when to importance resample: resampling occurs when
        ``ess < ess_threshold * num_particles``.
    """
    # TODO: Add window kwarg that defaults to float("inf")
    def __init__(self, model, guide, num_particles, max_plate_nesting, *,
                 ess_threshold=0.5):
        assert 0 < ess_threshold <= 1
        self.model = model
        self.guide = guide
        self.num_particles = num_particles
        self.max_plate_nesting = max_plate_nesting
        self.ess_threshold = ess_threshold

        # Equivalent to an empirical distribution, but allows a
        # user-defined dynamic collection of tensors.
        self.state = SMCState(self.num_particles)

    def init(self, *args, **kwargs):
        """
        Perform any initialization for sequential importance resampling.
        Any args or kwargs are passed to the model and guide
        """
        self.particle_plate = pyro.plate("particles", self.num_particles, dim=-1-self.max_plate_nesting)
        with poutine.block(), self.particle_plate:
            with self.state._lock():
                guide_trace = poutine.trace(self.guide.init).get_trace(self.state, *args, **kwargs)
            model = poutine.replay(self.model.init, guide_trace)
            model_trace = poutine.trace(model).get_trace(self.state, *args, **kwargs)

        self._update_weights(model_trace, guide_trace)
        self._maybe_importance_resample()

    def step(self, *args, **kwargs):
        """
        Take a filtering step using sequential importance resampling updating the
        particle weights and values while resampling if desired.
        Any args or kwargs are passed to the model and guide
        """
        with poutine.block(), self.particle_plate:
            with self.state._lock():
                guide_trace = poutine.trace(self.guide.step).get_trace(self.state, *args, **kwargs)
            model = poutine.replay(self.model.step, guide_trace)
            model_trace = poutine.trace(model).get_trace(self.state, *args, **kwargs)

        self._update_weights(model_trace, guide_trace)
        self._maybe_importance_resample()

    def get_empirical(self):
        """
        :returns: a marginal distribution over all state tensors.
        :rtype: a dictionary with keys which are latent variables and values
            which are :class:`~pyro.distributions.Empirical` objects.
        """
        return {key: dist.Empirical(value, self.state._log_weights)
                for key, value in self.state.items()}

    @torch.no_grad()
    def _update_weights(self, model_trace, guide_trace):
        # w_t <-w_{t-1}*p(y_t|z_t) * p(z_t|z_t-1)/q(z_t)

        model_trace = prune_subsample_sites(model_trace)
        guide_trace = prune_subsample_sites(guide_trace)

        model_trace.compute_log_prob()
        guide_trace.compute_log_prob()

        for name, guide_site in guide_trace.nodes.items():
            if guide_site["type"] == "sample":
                model_site = model_trace.nodes[name]
                log_p = model_site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                log_q = guide_site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                self.state._log_weights += log_p - log_q
                if not (self.state._log_weights.max() > -math.inf):
                    raise SMCFailed("Failed to find feasible hypothesis after site {}"
                                    .format(name))

        for site in model_trace.nodes.values():
            if site["type"] == "sample" and site["is_observed"]:
                log_p = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                self.state._log_weights += log_p
                if not (self.state._log_weights.max() > -math.inf):
                    raise SMCFailed("Failed to find feasible hypothesis after site {}"
                                    .format(site["name"]))

        self.state._log_weights -= self.state._log_weights.max()

    def _maybe_importance_resample(self):
        if not self.state:
            return
        # Decide whether to resample based on ESS.
        logp = self.state._log_weights
        logp -= logp.logsumexp(-1)
        probs = logp.exp()
        ess = probs.dot(probs).reciprocal()
        if ess < self.ess_threshold * self.num_particles:
            self._importance_resample(probs)

    def _importance_resample(self, probs):
        index = _systematic_sample(probs)
        self.state._resample(index)


def _systematic_sample(probs):
    # Systematic sampling preserves diversity better than multinomial sampling
    # via Categorical(probs).sample().
    batch_shape, size = probs.shape[:-1], probs.size(-1)
    n = probs.cumsum(-1).mul_(size).add_(torch.rand(batch_shape + (1,)))
    n = n.floor_().clamp_(min=0, max=size).long()
    diff = probs.new_zeros(batch_shape + (size + 1,))
    diff.scatter_add_(-1, n, torch.ones_like(probs))
    index = diff[..., :-1].cumsum(-1).long()
    return index


class SMCState(dict):
    """
    Dictionary-like object to hold a vectorized collection of tensors to
    represent all state during inference with :class:`SMCFilter`. During
    inference, the :class:`SMCFilter` resample these tensors.

    Keys may have arbitrary hashable type.
    Values must be :class:`torch.Tensor` s.

    :param int num_particles:
    """
    def __init__(self, num_particles):
        assert isinstance(num_particles, int) and num_particles > 0
        super().__init__()
        self._num_particles = num_particles
        self._log_weights = torch.zeros(num_particles)
        self._locked = False

    @contextlib.contextmanager
    def _lock(self):
        self._locked = True
        try:
            yield
        finally:
            self._locked = False

    def __setitem__(self, key, value):
        if self._locked:
            raise RuntimeError("Guide cannot write to SMCState")
        if is_validation_enabled():
            if not isinstance(value, torch.Tensor):
                raise TypeError("Only Tensors can be stored in an SMCState, but got {}"
                                .format(type(value).__name__))
            if value.dim() == 0 or value.size(0) != self._num_particles:
                raise ValueError("Expected leading dim of size {} but got shape {}"
                                 .format(self._num_particles, value.shape))
        super().__setitem__(key, value)

    def _resample(self, index):
        for key, value in self.items():
            self[key] = value[index].contiguous()
        self._log_weights.fill_(0.)
