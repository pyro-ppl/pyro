import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.poutine.util import prune_subsample_sites


def _extract_samples(trace):
    return {name: site["value"]
            for name, site in trace.nodes.items()
            if site["type"] == "sample"
            if not site["is_observed"]
            if type(site["fn"]).__name__ != "_Subsample"}


class SMCFilter(object):
    """
    :class:`SMCFilter` is the top-level interface for filtering via sequential
    monte carlo.

    The model and guide should be objects with two methods: ``.init()`` and
    ``.step()``. These two methods should have the same signature as :meth:`init`
    and :meth:`step` of this class. These methods are intended to be called first
    with :meth:`init`, then with :meth:`step` repeatedly.

    :param object model: probabilistic model defined as a function
    :param object guide: guide used for sampling defined as a function
    :param int num_particles: The number of particles used to form the
        distribution.
    :param int max_plate_nesting: Bound on max number of nested
        :func:`pyro.plate` contexts.
    """
    # TODO: Add window kwarg that defaults to float("inf")
    def __init__(self, model, guide, num_particles, max_plate_nesting):
        self.model = model
        self.guide = guide
        self.num_particles = num_particles
        self.max_plate_nesting = max_plate_nesting

        # Equivalent to an empirical distribution.
        self._values = {}
        self._log_weights = torch.zeros(self.num_particles)

    def init(self, *args, **kwargs):
        """
        Perform any initialization for sequential importance resampling.
        Any args or kwargs are passed to the model and guide
        """
        self.particle_plate = pyro.plate("particles", self.num_particles, dim=-1-self.max_plate_nesting)
        with poutine.block(), self.particle_plate:
            guide_trace = poutine.trace(self.guide.init).get_trace(*args, **kwargs)
            model = poutine.replay(self.model.init, guide_trace)
            model_trace = poutine.trace(model).get_trace(*args, **kwargs)

        self._update_weights(model_trace, guide_trace)
        self._values.update(_extract_samples(model_trace))
        self._maybe_importance_resample()

    def step(self, *args, **kwargs):
        """
        Take a filtering step using sequential importance resampling updating the
        particle weights and values while resampling if desired.
        Any args or kwargs are passed to the model and guide
        """
        with poutine.block(), self.particle_plate:
            guide_trace = poutine.trace(self.guide.step).get_trace(*args, **kwargs)
            model = poutine.replay(self.model.step, guide_trace)
            model_trace = poutine.trace(model).get_trace(*args, **kwargs)

        self._update_weights(model_trace, guide_trace)
        self._values.update(_extract_samples(model_trace))
        self._maybe_importance_resample()

    def get_values_and_log_weights(self):
        """
        Returns the particles and its (unnormalized) log weights.
        :returns: the values and unnormalized log weights.
        :rtype: tuple of dict and floats where the dict is a key of name of latent to value of latent.
        """
        # TODO: Be clear that these are unnormalized weights. May want to normalize later.
        return self._values, self._log_weights

    def get_empirical(self):
        """
        :returns: a marginal distribution over every latent variable.
        :rtype: a dictionary with keys which are latent variables and values
            which are :class:`~pyro.distributions.Empirical` objects.
        """
        return {name: dist.Empirical(value, self._log_weights)
                for name, value in self._values.items()}

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
                self._log_weights += log_p - log_q

        for site in model_trace.nodes.values():
            if site["type"] == "sample" and site["is_observed"]:
                log_p = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                self._log_weights += log_p

        self._log_weights -= self._log_weights.max()

    def _maybe_importance_resample(self):
        if True:  # TODO check perplexity
            self._importance_resample()

    def _importance_resample(self):
        # TODO: Turn quadratic algo -> linear algo by being lazier
        index = dist.Categorical(logits=self._log_weights).sample(sample_shape=(self.num_particles,))
        self._values = {name: value[index].contiguous() for name, value in self._values.items()}
        self._log_weights.fill_(0.)
