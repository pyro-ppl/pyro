# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import operator
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import ExitStack
from functools import reduce

import torch
from torch.distributions import biject_to, constraints

import pyro.distributions as dist
import pyro.distributions.hmm
import pyro.poutine as poutine
from pyro.distributions.transforms import DiscreteCosineTransform
from pyro.infer import MCMC, NUTS, SMCFilter, infer_discrete
from pyro.infer.autoguide import init_to_value
from pyro.infer.reparam import DiscreteCosineReparam
from pyro.util import warn_if_nan

from .util import cat2, clamp, quantize, quantize_enumerate

logger = logging.getLogger(__name__)


class CompartmentalModel(ABC):
    """
    Abstract base class for discrete-time discrete-value stochastic
    compartmental models.

    Derived classes must implement methods :meth:`heuristic`,
    :meth:`initialize`, :meth:`transition_fwd`, :meth:`transition_bwd`.
    Derived classes may optionally implement :meth:`global_model` and override
    the ``series`` attribute.

    Example usage::

        # First implement a concrete derived class.
        class MyModel(CompartmentalModel):
            def __init__(self, ...): ...
            def heuristic(self): ...
            def global_model(self): ...
            def initialize(self, params): ...
            def transition_fwd(self, params, state, t): ...
            def transition_bwd(self, params, prev, curr, t): ...

        # Run inference to fit the model to data.
        model = MyModel(...)
        model.fit(num_samples=100)
        R0 = model.samples["R0"]  # An example parameter.
        print("R0 = {:0.3g} \u00B1 {:0.3g}".format(R0.mean(), R0.std()))

        # Predict latent variables.
        samples = model.predict()

        # Forecast forward.
        samples = model.predict(forecast=30)

        # You can assess future interventions (applied after ``duration``) by
        # storing them as attributes that are read by your derived methods.
        model.my_intervention = False
        samples1 = model.predict(forecast=30)
        model.my_intervention = True
        samples2 = model.predict(forecast=30)
        effect = samples2["my_result"].mean() - samples1["my_result"].mean()
        print("average effect = {:0.3g}".format(effect))

    :cvar tuple series: Tuple of names of time series names, in addition to
        ``self.compartments``. These will be concatenated along the time axis
        in returned sample dictionaries.
    :ivar dict samples: Dictionary of posterior samples.
    :param list compartments: A list of strings of compartment names.
    :param int duration:
    :param int population:
    """

    def __init__(self, compartments, duration, population, *,
                 num_quant_bins=4):
        super().__init__()

        assert isinstance(duration, int)
        assert duration >= 1
        self.duration = duration

        if isinstance(population, torch.Tensor):
            assert population.dim() == 1
            assert (population >= 1).all()
            self.is_regional = True
            if self.max_plate_nesting == 0:
                self.max_plate_nesting = 1
        else:
            assert isinstance(population, int)
            assert population >= 2
            self.is_regional = False
        self.population = population

        compartments = tuple(compartments)
        assert all(isinstance(name, str) for name in compartments)
        assert len(compartments) == len(set(compartments))
        self.compartments = compartments

        # Inference state.
        self.samples = {}
        self._clear_plates()

    @property
    def region_plate(self):
        """
        Either a ``pyro.plate`` or a trivial ``ExitStack`` depending on whether
        this model ``.is_regional``.
        """
        if self._region_plate is None:
            if self.is_regional:
                self._region_plate = pyro.plate("region", len(self.population), dim=-1)
            else:
                self._region_plate = ExitStack()
        return self._region_plate

    def _clear_plates(self):
        self._region_plate = None

    # Overridable attributes and methods ########################################

    max_plate_nesting = 0
    series = ()
    full_mass = False

    @torch.no_grad()
    def heuristic(self, num_particles=1024):
        """
        Finds an initial feasible guess of all latent variables, consistent
        with observed data. This is needed because not all hypotheses are
        feasible and HMC needs to start at a feasible solution to progress.

        The default implementation attempts to find a feasible state using
        :class:`~pyro.infer.smcfilter.SMCFilter` with proprosals from the
        prior.  However this method may be overridden in cases where SMC
        performs poorly e.g. in high-dimensional models.

        :param int num_particles: Number of particles used for SMC.
        :returns: A dictionary mapping sample site name to tensor value.
        :rtype: dict
        """
        # Run SMC.
        model = _SMCModel(self)
        guide = _SMCGuide(self)
        smc = SMCFilter(model, guide, num_particles=num_particles,
                        max_plate_nesting=self.max_plate_nesting)
        smc.init()
        for t in range(1, self.duration):
            smc.step()

        # Select the most probable hypothesis.
        i = int(smc.state._log_weights.max(0).indices)
        init = {key: value[i] for key, value in smc.state.items()}

        # Fill in sample site values.
        init = self.generate(init)
        aux = torch.stack([init[name] for name in self.compartments], dim=0)
        init["auxiliary"] = clamp(aux, min=0.5, max=self.population - 0.5)
        return init

    def global_model(self):
        """
        Samples and returns any global parameters.

        :returns: An arbitrary object of parameters (e.g. ``None``).
        """
        return None

    # TODO Allow stochastic initialization.
    @abstractmethod
    def initialize(self, params):
        """
        Returns initial counts in each compartment.

        :param params: The global params returned by :meth:`global_model`.
        :returns: A dict mapping compartment name to initial value.
        :rtype: dict
        """
        raise NotImplementedError

    @abstractmethod
    def transition_fwd(self, params, state, t):
        """
        Forward generative process for dynamics.

        This inputs a current ``state`` and stochastically updates that
        state in-place.

        Note that this method is called under two different interpretations.
        During :meth:`generate` this is called to generate a single sample.
        During :meth:`predict` thsi is called to forecast a batch of samples,
        conditioned on posterior samples for the time interval
        ``[0:duration]``.

        :param params: The global params returned by :meth:`global_model`.
        :param dict state: A dictionary mapping compartment name to current
            tensor value. This should be updated in-place.
        :param int t: Time index.
        """
        raise NotImplementedError

    @abstractmethod
    def transition_bwd(self, params, prev, curr, t):
        """
        Backward factor graph for dynamics.

        This inputs hypotheses for two subsequent time steps
        (``prev``,``curr``) and makes observe statements
        ``pyro.sample(..., obs=...)`` to declare probability factors.

        Note that this method is called under two different interpretations.
        During inference it is called vectorizing over time but with a single
        sample. During prediction it is called sequentially for each time
        step, but always vectorizing over samples.

        :param params: The global params returned by :meth:`global_model`.
        :param dict prev: A dictionary mapping compartment name to previous
            tensor value. This should not be modified.
        :param dict curr: A dictionary mapping compartment name to current
            tensor value. This should not be modified.
        :param t: A time-like index. During inference ``t`` will be
            an indexing tuple that reshapes data tensors. During prediction
            ``t`` will be an actual integer time index.
        """
        raise NotImplementedError

    # Inference interface ########################################

    @torch.no_grad()
    def generate(self, fixed={}):
        """
        Generate data from the prior.

        :pram dict fixed: A dictionary of parameters on which to condition.
            These must be top-level parentless nodes, i.e. have no
            upstream stochastic dependencies.
        :returns: A dictionary mapping sample site name to sampled value.
        :rtype: dict
        """
        fixed = {k: torch.as_tensor(v) for k, v in fixed.items()}
        model = self._generative_model
        model = poutine.condition(model, fixed)
        trace = poutine.trace(model).get_trace()
        samples = OrderedDict((name, site["value"])
                              for name, site in trace.nodes.items()
                              if site["type"] == "sample")

        self._concat_series(samples)
        return samples

    def fit(self, **options):
        r"""
        Runs inference to generate posterior samples.

        This uses the :class:`~pyro.infer.mcmc.nuts.NUTS` kernel to run
        :class:`~pyro.infer.mcmc.api.MCMC`, setting the ``.samples``
        attribute on completion.

        :param \*\*options: Options passed to
            :class:`~pyro.infer.mcmc.api.MCMC`. The remaining options are
            pulled out and have special meaning.
        :param int max_tree_depth: (Default 5). Max tree depth of the
            :class:`~pyro.infer.mcmc.nuts.NUTS` kernel.
        :param full_mass: (Default ``False``). Specification of mass matrix
            of the :class:`~pyro.infer.mcmc.nuts.NUTS` kernel.
        :param int num_quant_bins: The number of quantization bins to use. Note
            that computational cost is exponential in `num_quant_bins`.
            Defaults to 4.
        :param float dct: If provided, use a discrete cosine reparameterizer
            with this value as smoothness.
        :param int heuristic_num_particles: Passed to :meth:`heuristic` as
            ``num_particles``. Defaults to 1024.
        :returns: An MCMC object for diagnostics, e.g. ``MCMC.summary()``.
        :rtype: ~pyro.infer.mcmc.api.MCMC
        """
        # Save these options for .predict().
        self.num_quant_bins = options.pop("num_quant_bins", 4)
        self._dct = options.pop("dct", None)

        # Heuristically initialze to feasible latents.
        logger.info("Heuristically initializing...")
        heuristic_options = {k.replace("heuristic_", ""): options.pop(k)
                             for k in list(options) if k.startswith("heuristic_")}
        init_values = self.heuristic(**heuristic_options)
        assert isinstance(init_values, dict)
        assert "auxiliary" in init_values, \
            ".heuristic() did not define auxiliary value"
        if self._dct is not None:
            # Also initialize DCT transformed coordinates.
            x = init_values["auxiliary"]
            x = biject_to(constraints.interval(-0.5, self.population + 0.5)).inv(x)
            x = DiscreteCosineTransform(smooth=self._dct)(x)
            init_values["auxiliary_dct"] = x

        # Configure a kernel.
        logger.info("Running inference...")
        max_tree_depth = options.pop("max_tree_depth", 5)
        full_mass = options.pop("full_mass", self.full_mass)
        model = self._vectorized_model
        if self._dct is not None:
            rep = DiscreteCosineReparam(smooth=self._dct)
            model = poutine.reparam(model, {"auxiliary": rep})
        kernel = NUTS(model,
                      full_mass=full_mass,
                      init_strategy=init_to_value(values=init_values),
                      max_tree_depth=max_tree_depth)

        # Run mcmc.
        mcmc = MCMC(kernel, **options)
        mcmc.run()
        self.samples = mcmc.get_samples()
        return mcmc  # E.g. so user can run mcmc.summary().

    @torch.no_grad()
    def predict(self, forecast=0):
        """
        Predict latent variables and optionally forecast forward.

        This may be run only after :meth:`fit` and draws the same
        ``num_samples`` as passed to :meth:`fit`.

        :param int forecast: The number of time steps to forecast forward.
        :returns: A dictionary mapping sample site name (or compartment name)
            to a tensor whose first dimension corresponds to sample batching.
        :rtype: dict
        """
        if not self.samples:
            raise RuntimeError("Missing samples, try running .fit() first")
        samples = self.samples
        num_samples = len(next(iter(samples.values())))
        particle_plate = pyro.plate("particles", num_samples,
                                    dim=-1 - self.max_plate_nesting)

        # Sample discrete auxiliary variables conditioned on the continuous
        # variables sampled by _vectorized_model. This samples only time steps
        # [0:duration]. Here infer_discrete runs a forward-filter
        # backward-sample algorithm.
        logger.info("Predicting latent variables for {} time steps..."
                    .format(self.duration))
        model = self._sequential_model
        model = poutine.condition(model, samples)
        model = particle_plate(model)
        if self._dct is not None:
            # Apply the same reparameterizer as during inference.
            rep = DiscreteCosineReparam(smooth=self._dct)
            model = poutine.reparam(model, {"auxiliary": rep})
        model = infer_discrete(model, first_available_dim=-2 - self.max_plate_nesting)
        trace = poutine.trace(model).get_trace()
        samples = OrderedDict((name, site["value"])
                              for name, site in trace.nodes.items()
                              if site["type"] == "sample")

        # Optionally forecast with the forward _generative_model. This samples
        # time steps [duration:duration+forecast].
        if forecast:
            logger.info("Forecasting {} steps ahead...".format(forecast))
            model = self._generative_model
            model = poutine.condition(model, samples)
            model = particle_plate(model)
            trace = poutine.trace(model).get_trace(forecast)
            samples = OrderedDict((name, site["value"])
                                  for name, site in trace.nodes.items()
                                  if site["type"] == "sample")

        self._concat_series(samples, forecast)
        return samples

    # Internal helpers ########################################

    def _concat_series(self, samples, forecast=0):
        """
        Concatenate sequential time series into tensors, in-place.

        :param dict samples: A dictionary of samples.
        """
        dim = -2 if self.is_regional else -1
        for name in self.compartments + self.series:
            pattern = name + "_[0-9]+"
            series = []
            for key in list(samples):
                if re.match(pattern, key):
                    series.append(samples.pop(key))
            if series:
                assert len(series) == self.duration + forecast
                series = torch.broadcast_tensors(*map(torch.as_tensor, series))
                samples[name] = torch.stack(series, dim=dim)

    def _generative_model(self, forecast=0):
        """
        Forward generative model used for simulation and forecasting.
        """
        # Sample global parameters.
        params = self.global_model()

        # Sample initial values.
        state = self.initialize(params)
        state = {k: v if isinstance(v, torch.Tensor) else torch.tensor(float(v))
                 for k, v in state.items()}

        # Sequentially transition.
        for t in range(self.duration + forecast):
            self.transition_fwd(params, state, t)
            with self.region_plate:
                for name in self.compartments:
                    pyro.deterministic("{}_{}".format(name, t), state[name])

        self._clear_plates()

    def _sequential_model(self):
        """
        Sequential model used to sample latents in the interval [0:duration].
        """
        C = len(self.compartments)
        T = self.duration
        R_shape = getattr(self.population, "shape", ())  # Region shape.

        # Sample global parameters.
        params = self.global_model()

        # Sample the continuous reparameterizing variable.
        shape = (C, T) + R_shape
        auxiliary = pyro.sample("auxiliary",
                                dist.Uniform(-0.5, self.population + 0.5)
                                    .mask(False).expand(shape).to_event(len(shape)))
        if self.is_regional:
            # This reshapes from (particle, 1, C, T, R) -> (particle, C, T, R)
            # to allow aux below to have shape (particle, R) for region_plate.
            auxiliary = auxiliary.squeeze(-4)

        # Sequentially transition.
        curr = self.initialize(params)
        for t, aux_t in poutine.markov(enumerate(auxiliary.unbind(2))):
            with self.region_plate:
                prev, curr = curr, {}
                for name, aux in zip(self.compartments, aux_t.unbind(1)):
                    curr[name] = quantize("{}_{}".format(name, t), aux,
                                          min=0, max=self.population,
                                          num_quant_bins=self.num_quant_bins)
                    # In regional models, enable approximate inference by using aux
                    # as a non-enumerated proxy for enumerated compartment values.
                    if self.is_regional:
                        curr[name + "_approx"] = aux
                        prev.setdefault(name + "_approx", prev[name])
            self.transition_bwd(params, prev, curr, t)

        self._clear_plates()

    def _vectorized_model(self):
        """
        Vectorized model used for inference.
        """
        C = len(self.compartments)
        T = self.duration
        Q = self.num_quant_bins
        R_shape = getattr(self.population, "shape", ())  # Region shape.

        # Sample global parameters.
        params = self.global_model()

        # Sample the continuous reparameterizing variable.
        shape = (C, T) + R_shape
        auxiliary = pyro.sample("auxiliary",
                                dist.Uniform(-0.5, self.population + 0.5)
                                    .mask(False).expand(shape).to_event(len(shape)))
        assert auxiliary.shape == shape, "particle plates are not supported"

        # Manually enumerate.
        curr, logp = quantize_enumerate(auxiliary, min=0, max=self.population,
                                        num_quant_bins=self.num_quant_bins)
        curr = OrderedDict(zip(self.compartments, curr))
        logp = OrderedDict(zip(self.compartments, logp))

        # Truncate final value from the right then pad initial value onto the left.
        init = self.initialize(params)
        prev = {}
        for name in self.compartments:
            value = init[name]
            if isinstance(value, torch.Tensor):
                value = value[..., None]  # Because curr is enumerated on the right.
            prev[name] = cat2(value, curr[name][:-1], dim=-3 if self.is_regional else -2)

        # Reshape to support broadcasting, similar to EnumMessenger.
        def enum_reshape(tensor, position):
            assert tensor.size(-1) == Q
            assert tensor.dim() <= self.max_plate_nesting + 2
            tensor = tensor.permute(tensor.dim() - 1, *range(tensor.dim() - 1))
            shape = [Q] + [1] * (position + self.max_plate_nesting - (tensor.dim() - 2))
            shape.extend(tensor.shape[1:])
            return tensor.reshape(shape)

        for e, name in enumerate(self.compartments):
            curr[name] = enum_reshape(curr[name], e)
            logp[name] = enum_reshape(logp[name], e)
            prev[name] = enum_reshape(prev[name], C + e)

        # In regional models, enable approximate inference by using aux
        # as a non-enumerated proxy for enumerated compartment values.
        if self.is_regional:
            for name, aux in zip(self.compartments, auxiliary):
                curr[name + "_approx"] = aux
                prev[name + "_approx"] = cat2(init[name], aux[:-1],
                                              dim=-2 if self.is_regional else -1)

        # Record transition factors.
        with poutine.block(), poutine.trace() as tr:
            with pyro.plate("time", T, dim=-1 - self.max_plate_nesting):
                t = slice(None)  # Used to slice data tensors.
                self.transition_bwd(params, prev, curr, t)
        tr.trace.compute_log_prob()
        for name, site in tr.trace.nodes.items():
            if site["type"] == "sample":
                logp[name] = site["log_prob"]

        # Manually perform variable elimination.
        logp = reduce(operator.add, logp.values())
        logp = logp.reshape(Q ** C, Q ** C, T, -1)  # prev, curr, T, batch
        logp = logp.permute(3, 2, 0, 1).squeeze(0)  # batch, T, prev, curr
        logp = pyro.distributions.hmm._sequential_logmatmulexp(logp)  # batch, prev, curr
        logp = logp.reshape(-1, Q ** C * Q ** C).logsumexp(-1).sum()
        warn_if_nan(logp)
        pyro.factor("transition", logp)

        self._clear_plates()


class _SMCModel:
    """
    Helper to initialize a CompartmentalModel to a feasible initial state.
    """
    def __init__(self, model):
        assert isinstance(model, CompartmentalModel)
        self.model = model

    def init(self, state):
        with poutine.trace() as tr:
            params = self.model.global_model()
        for name, site in tr.trace.nodes.items():
            if site["type"] == "sample":
                state[name] = site["value"]

        self.t = 0
        state.update(self.model.initialize(params))
        self.step(state)  # Take one step since model.initialize is deterministic.

    def step(self, state):
        with poutine.block(), poutine.condition(data=state):
            params = self.model.global_model()
        with poutine.trace() as tr:
            self.model.transition_fwd(params, state, self.t)
        for name, site in tr.trace.nodes.items():
            if site["type"] == "sample" and not site["is_observed"]:
                state[name] = site["value"]
        self.t += 1


class _SMCGuide(_SMCModel):
    """
    Like _SMCModel but does not update state and does not observe.
    """
    def init(self, state):
        super().init(state.copy())

    def step(self, state):
        with poutine.block(hide_types=["observe"]):
            super().step(state.copy())
