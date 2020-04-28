# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro
import pyro.distributions as dist

from .compartmental import CompartmentalModel
from .distributions import infection_dist


class SimpleSIRModel(CompartmentalModel):
    """
    Susceptible-Infected-Recovered model.

    To customize this model we recommend forking and editing this class.

    This is a stochastic discrete-time discrete-state model with three
    compartments: "S" for susceptible, "I" for infected, and "R" for
    recovered individuals (the recovered individuals are implicit: ``R =
    population - S - I``) with transitions ``S -> I -> R``.

    :param int population: Total ``population = S + I + R``.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param iterable data: Time series of new observed infections.
    :param int data: Time series of new observed infections, i.e. a Binomial
        subset of the ``S -> I`` transitions at each time step.
    """

    def __init__(self, population, recovery_time, data):
        compartments = ("S", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data

    series = ("S2I", "I2R", "obs")
    full_mass = [("R0", "rho")]

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Uniform(0, 1))
        return R0, tau, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "I": 1}

    def transition_fwd(self, params, state, t):
        R0, tau, rho = params

        # Sample flows between compartments.
        S2I = pyro.sample("S2I_{}".format(t),
                          infection_dist(individual_rate=R0 / tau,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population))
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], 1 / tau))

        # Update compartments with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t] if t < self.duration else None)

    def transition_bwd(self, params, prev, curr, t):
        R0, tau, rho = params

        # Reverse the flow computation.
        S2I = prev["S"] - curr["S"]
        I2R = prev["I"] - curr["I"] + S2I

        # Condition on flows between compartments.
        pyro.sample("S2I_{}".format(t),
                    infection_dist(individual_rate=R0 / tau,
                                   num_susceptible=prev["S"],
                                   num_infectious=prev["I"],
                                   population=self.population),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(prev["I"], 1 / tau),
                    obs=I2R)

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t])


class OverdispersedSIRModel(CompartmentalModel):
    """
    Overdispersed Susceptible-Infected-Recovered model.

    To customize this model we recommend forking and editing this class.

    This is a stochastic discrete-time discrete-state model with three
    compartments: "S" for susceptible, "I" for infected, and "R" for
    recovered individuals (the recovered individuals are implicit: ``R =
    population - S - I``) with transitions ``S -> I -> R``.

    This model accounts for superspreading (overdispersed individual
    reproductive number) by assuming each infected individual infects
    BetaBinomial-many susceptible individuals, where the BetaBinomial
    distribution acts as an overdispersed Binomial distribution, adapting the
    more standard NegativeBinomial distribution that acts as an overdispersed
    Poisson distribution [1,2] to the setting of finite populations. To
    preserve Markov structure, we follow [2] and assume all infections by a
    single individual occur on the single time step where that individual makes
    an ``I -> R`` transition.

    **References**

    [1] J. O. Lloyd-Smith, S. J. Schreiber, P. E. Kopp, W. M. Getz (2005)
        "Superspreading and the effect of individual variation on disease
        emergence"
        https://www.nature.com/articles/nature04153.pdf
    [2] Lucy M. Li, Nicholas C. Grassly, Christophe Fraser (2017)
        "Quantifying Transmission Heterogeneity Using Both Pathogen Phylogenies
        and Incidence Time Series"
        https://academic.oup.com/mbe/article/34/11/2982/3952784

    :param int population: Total ``population = S + I + R``.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param iterable data: Time series of new observed infections.
    :param int data: Time series of new observed infections, i.e. a Binomial
        subset of the ``S -> I`` transitions at each time step.
    """

    def __init__(self, population, recovery_time, data):
        compartments = ("S", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data

    series = ("S2I", "I2R", "obs")
    full_mass = [("R0", "rho", "k")]

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        k = pyro.sample("k", dist.Exponential(1.))
        rho = pyro.sample("rho", dist.Uniform(0, 1))
        return R0, k, tau, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "I": 1}

    def transition_fwd(self, params, state, t):
        R0, k, tau, rho = params

        # Sample flows between compartments.
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], 1 / tau))
        S2I = pyro.sample("S2I_{}".format(t),
                          infection_dist(individual_rate=R0,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population,
                                         concentration=k))

        # Update compartments with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t] if t < self.duration else None)

    def transition_bwd(self, params, prev, curr, t):
        R0, k, tau, rho = params

        # Reverse the flow computation.
        S2I = prev["S"] - curr["S"]
        I2R = prev["I"] - curr["I"] + S2I

        # Condition on flows between compartments.
        pyro.sample("S2I_{}".format(t),
                    infection_dist(individual_rate=R0,
                                   num_susceptible=prev["S"],
                                   num_infectious=prev["I"],
                                   population=self.population,
                                   concentration=k),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(prev["I"], 1 / tau),
                    obs=I2R)

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t])
