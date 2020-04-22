# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import logging

import torch
from torch.nn.functional import pad

import pyro.distributions as dist
import pyro.distributions.hmm
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, config_enumerate, infer_discrete
from pyro.infer.autoguide import init_to_value
from pyro.infer.reparam import DiscreteCosineReparam
from pyro.util import warn_if_nan

from .util import quantize, quantize_enumerate

logger = logging.getLogger(__name__)


class CompartmentalModel(ABC):
    """
    Abstract base class for discrete-time discrete-value stochastic
    compartmental models.

    Derived classes must implement methods :method:`heuristic`,
    :method:`initialize`, :method:`transition_fwd`, :method:`transition_bwd`.
    Derived classes may optionally implement :method:`global_model`.

    Example usage::

        # First implement a concrete derived class.
        class MyModel(CompartmentalModel):
            def __init__(self, ...): ...
            def heuristic(self): ...
            def global_model(self): ...
            def initialize(self, params): ...
            def transition_fwd(self, params, state, t): ...
            def transition_bwd(self, params, prev, curr): ...

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

    :ivar dict samples: Dictionary of posterior samples.
    :param list compartments: A list of strings of compartment names.
    :param int duration:
    :param int population:
    """
    def __init__(self, compartments, duration, population):
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

    # Abstract methods ########################################

    @abstractmethod
    def heuristic(self):
        raise NotImplementedError

    def global_model(self):
        return None

    @abstractmethod
    def initialize(self, params):
        raise NotImplementedError

    @abstractmethod
    def transition_fwd(self, params, state, t):
        raise NotImplementedError

    @abstractmethod
    def transition_bwd(self, params, prev, curr):
        raise NotImplementedError

    # Inference interface ########################################

    def validate(self):
        """
        Validate agreement between :meth:`transition_fwd` and
        :meth:`transition_bwd`.
        """
        raise NotImplementedError("TODO")

    def generate(self):
        raise NotImplementedError("TODO")

    def fit(self, **options):
        self._dct = options.pop("dct", None)

        model = self._vectorized_model
        if self._dct is not None:
            rep = DiscreteCosineReparam(smooth=self._dct)
            model = poutine.reparam(model, {"auxiliary": rep})

        # Configure a kernel.
        max_tree_depth = options.pop("max_tree_depth", 5)
        full_mass = options.pop("full_mass", False)
        kernel = NUTS(self._vectorized_model,
                      full_mass=full_mass,
                      max_tree_depth=max_tree_depth)

        # Run mcmc.
        mcmc = MCMC(kernel, **options)
        mcmc.run()

        self.samples = mcmc.get_samples()
        self._max_plate_nesting = kernel._max_plate_nesting
        return mcmc

    @torch.no_grad()
    def predict(self, forecast=0):
        if not self.samples:
            raise RuntimeError("Missing samples, try running .fit() first")
        samples = self.samples
        num_samples = len(next(iter(samples.values())))
        particle_plate = pyro.plate("particles", num_samples, dim=-1)

        # Sample discrete auxiliary variables conditioned on the continuous
        # variables sampled in vectorized_model. This samples only time steps
        # [0:duration]. Here infer_discrete runs a forward-filter
        # backward-sample algorithm.
        logger.info("Predicting latent variables for {} time steps..."
                    .format(self.duration))
        model = poutine.condition(continuous_model, samples)
        model = particle_plate(model)
        if self._dct is not None:
            # Apply the same reparameterizer as during inference.
            rep = DiscreteCosineReparam(smooth=self._dct)
            model = poutine.reparam(model, {"auxiliary": rep})
        model = infer_discrete(model, first_available_dim=-1 - self._max_plate_nesting)
        trace = poutine.trace(model).get_trace()
        samples = {name: site["value"]
                   for name, site in trace.nodes.items()
                   if site["type"] == "sample"}

        # Optionally forecast with the forward _generative_model. This samples
        # time steps [duration:duration+forecast].
        if forecast:
            logger.info("Forecasting {} steps ahead...".format(forecast))
            model = poutine.condition(discrete_model, samples)
            model = particle_plate(model)
            trace = poutine.trace(model).get_trace()
            samples = {name: site["value"]
                       for name, site in trace.nodes.items()
                       if site["type"] == "sample"}

        # Concatenate sequential time series into tensors.
        for key in self.compartments:
            pattern = key + "_[0-9]+"
            series = [value
                      for name, value in samples.items()
                      if re.match(pattern, name)]
            assert len(series) == self.duration + forecast
            series[0] = series[0].expand(series[1].shape)
            samples[key] = torch.stack(series, dim=-1)

        return samples

    # Internal helpers ########################################

    def _generative_model(self):
        # Sample global parameters.
        params = self.global_model()

        # Sample initial values.
        state = self.initialize(params)
        state = {i: torch.tensor(value) for i, value in state.items()}

        # Sequentially transition.
        for t  in range(self.duration):
            self.transition_fwd(params, state, t)
            for name in self.compartments:
                pyro.deterministic("{}_{}".format(name, t), state[name])

    def _sequential_model(self):
        # Sample global parameters.
        params = global_model(population)

        # Sample the continuous reparameterizing variable.
        auxiliary = pyro.sample("auxiliary",
                                dist.Uniform(-0.5, population + 0.5)
                                    .mask(False)
                                    .expand(len(self.compartments), self.duration)
                                    .to_event(2))

        # Sample initial values.
        curr = self.initialize(params)

        # Sequentially transition.
        for t, datum in poutine.markov(enumerate(data)):
            prev = curr
            curr = quantize("state_{}".format(t), auxiliary[..., t],
                            min=0, max=self.population)
            curr = dict(zip(self.compartments, curr.unbind(-2)))

            logp = self.transition_bwd(params, prev, curr)
            pyro.factor("transition_{}".format(t), logp)

    def _vectorized_model(self):
        # Sample global parameters.
        params = global_model(population)

        # Sample the continuous reparameterizing variable.
        auxiliary = pyro.sample("auxiliary",
                                dist.Uniform(-0.5, population + 0.5)
                                    .mask(False)
                                    .expand(len(self.compartments), self.duration)
                                    .to_event(2))

        # Sample initial values.
        init = self.initialize(params)

        # Manually enumerate.
        curr, curr_logp = quantize_enumerate(auxiliary, min=0, max=self.population)
        curr = dict(zip(self.compartments, curr.unbind(-2)))
        curr_logp = dict(zip(self.compartments, curr_logp.unbind(-2)))
        # Truncate final value from the right then pad initial value onto the left.
        prev = {}
        for i in self.compartments:
            if isinstance(init[i], int):
                prev[i] = pad(curr[i][:-1], (0, 0, 1, 0), value=init[i])
            else:
                raise NotImplementedError("TODO use torch.cat()")
        # Reshape to support broadcasting, similar EnumMessenger.
        T = self.duration
        Q = 4  # Number of quantization points.
        ########################################
        # TODO generalize this
        S_prev = S_prev.reshape(T, Q, 1, 1, 1)
        I_prev = I_prev.reshape(T, 1, Q, 1, 1)
        S_curr = S_curr.reshape(T, 1, 1, Q, 1)
        S_logp = S_logp.reshape(T, 1, 1, Q, 1)
        I_curr = I_curr.reshape(T, 1, 1, 1, Q)
        I_logp = I_logp.reshape(T, 1, 1, 1, Q)
        data = data.reshape(T, 1, 1, 1, 1)
        ########################################

        # Manually perform variable elimination.
        logp = S_logp + I_logp + self.transition(params, prev, curr)
        logp = logp.reshape(-1, Q ** len(self.compartments), Q ** len(self.compartments))
        logp = pyro.distributions.hmm._sequential_logmatmulexp(logp)
        logp = logp.reshape(-1).logsumexp(0)
        warn_if_nan(logp)
        pyro.factor("transition", logp)
