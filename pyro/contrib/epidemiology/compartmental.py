# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from abc import ABC, abstractmethod
from collections import OrderedDict

import torch
from torch.distributions import biject_to, constraints
from torch.nn.functional import pad

import pyro.distributions as dist
import pyro.distributions.hmm
import pyro.poutine as poutine
from pyro.distributions.transforms import DiscreteCosineTransform
from pyro.infer import MCMC, NUTS, SMCFilter, infer_discrete
from pyro.infer.autoguide import init_to_value
from pyro.infer.reparam import DiscreteCosineReparam
from pyro.util import warn_if_nan

from .util import quantize, quantize_enumerate

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

        assert isinstance(population, int)
        assert population >= 2
        self.population = population

        compartments = tuple(compartments)
        assert all(isinstance(name, str) for name in compartments)
        assert len(compartments) == len(set(compartments))
        self.compartments = compartments

        # Inference state.
        self.samples = {}

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

        # Select the most probably hypothesis.
        i = int(smc.state._log_weights.max(0).indices)
        init = {key: value[i] for key, value in smc.state.items()}

        # Fill in sample site values.
        init = self.generate(init)
        init["auxiliary"] = torch.stack([init[name] for name in self.compartments])
        init["auxiliary"].clamp_(min=0.5, max=self.population - 0.5)
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
        for name in self.compartments + self.series:
            pattern = name + "_[0-9]+"
            series = []
            for key in list(samples):
                if re.match(pattern, key):
                    series.append(samples.pop(key))
            if series:
                assert len(series) == self.duration + forecast
                series = torch.broadcast_tensors(*series)
                samples[name] = torch.stack(series, dim=-1)

    def _generative_model(self, forecast=0):
        """
        Forward generative model used for simulation and forecasting.
        """
        # Sample global parameters.
        params = self.global_model()

        # Sample initial values.
        state = self.initialize(params)
        state = {i: torch.tensor(float(value)) for i, value in state.items()}

        # Sequentially transition.
        for t in range(self.duration + forecast):
            self.transition_fwd(params, state, t)
            for name in self.compartments:
                pyro.deterministic("{}_{}".format(name, t), state[name])

    def _sequential_model(self):
        """
        Sequential model used to sample latents in the interval [0:duration].
        """
        # Sample global parameters.
        params = self.global_model()

        # Sample the continuous reparameterizing variable.
        auxiliary = pyro.sample("auxiliary",
                                dist.Uniform(-0.5, self.population + 0.5)
                                    .mask(False)
                                    .expand([len(self.compartments), self.duration])
                                    .to_event(2))

        # Sequentially transition.
        curr = self.initialize(params)
        for t in poutine.markov(range(self.duration)):
            aux_t = auxiliary[..., t]
            prev = curr
            curr = {name: quantize("{}_{}".format(name, t), aux,
                                   min=0, max=self.population,
                                   num_quant_bins=self.num_quant_bins)
                    for name, aux in zip(self.compartments, aux_t.unbind(-1))}
            self.transition_bwd(params, prev, curr, t)

    def _vectorized_model(self):
        """
        Vectorized model used for inference.
        """
        # Sample global parameters.
        params = self.global_model()

        # Sample the continuous reparameterizing variable.
        auxiliary = pyro.sample("auxiliary",
                                dist.Uniform(-0.5, self.population + 0.5)
                                    .mask(False)
                                    .expand([len(self.compartments), self.duration])
                                    .to_event(2))

        # Manually enumerate.
        curr, logp = quantize_enumerate(auxiliary, min=0, max=self.population,
                                        num_quant_bins=self.num_quant_bins)
        curr = OrderedDict(zip(self.compartments, curr))
        logp = OrderedDict(zip(self.compartments, logp))

        # Truncate final value from the right then pad initial value onto the left.
        init = self.initialize(params)
        prev = {}
        for name in self.compartments:
            if not isinstance(init[name], int):
                raise NotImplementedError("TODO use torch.cat()")
            prev[name] = pad(curr[name][:-1], (0, 0, 1, 0), value=init[name])

        # Reshape to support broadcasting, similar to EnumMessenger.
        C = len(self.compartments)
        T = self.duration
        Q = self.num_quant_bins  # Number of quantization points.

        def enum_shape(position):
            shape = [T] + [1] * (2 * C)
            shape[1 + position] = Q
            return torch.Size(shape)

        for e, name in enumerate(self.compartments):
            prev[name] = prev[name].reshape(enum_shape(e))
            curr[name] = curr[name].reshape(enum_shape(C + e))
            logp[name] = logp[name].reshape(enum_shape(C + e))
        t = (Ellipsis,) + (None,) * (2 * C)  # Used to unsqueeze data tensors.

        # Record transition factors.
        with poutine.block(), poutine.trace() as tr:
            self.transition_bwd(params, prev, curr, t)
        for name, site in tr.trace.nodes.items():
            if site["type"] == "sample":
                logp[name] = site["fn"].log_prob(site["value"])

        # Manually perform variable elimination.
        logp = sum(logp.values())
        logp = logp.reshape(T, Q ** C, Q ** C)
        logp = pyro.distributions.hmm._sequential_logmatmulexp(logp)
        logp = logp.reshape(-1).logsumexp(0)
        warn_if_nan(logp)
        pyro.factor("transition", logp)


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
