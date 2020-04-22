from abc import ABC, abstractmethod

import torch

import pyro.distributions as dist


class CompartmentalModel(ABC):
    """
    :param list compartments: A list of strings of compartment names.
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
        Validate agreement between :met:`transition_fwd` and
        :meth:`transition_bwd`.
        """
        raise NotImplementedError("TODO")

    def generate(self):
        raise NotImplementedError("TODO")

    def fit(self, **options):
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
        return mcmc

    def predict(self, forecast_steps=0):
        raise NotImplementedError("TODO")

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
            curr = quantize("state_{}".format(t), auxiliary[..., t], max=self.population)
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
