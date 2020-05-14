# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro
import pyro.distributions as dist

from .compartmental import CompartmentalModel
from .distributions import infection_dist


class SimpleSEIRModel(CompartmentalModel):
    """
    Susceptible-Exposed-Infected-Recovered model.

    To customize this model we recommend forking and editing this class.

    This is a stochastic discrete-time discrete-state model with four
    compartments: "S" for susceptible, "E" for exposed, "I" for infected,
    and "R" for recovered individuals (the recovered individuals are
    implicit: ``R = population - S - E - I``) with transitions
    ``S -> E -> I -> R``.

    :param int population: Total ``population = S + E + I + R``.
    :param float incubation_time: Mean incubation time (duration in state
        ``E``). Must be greater than 1.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param iterable data: Time series of new observed infections. Each time
        step is Binomial distributed between 0 and the number of ``S -> E``
        transitions. This allows false negative but no false positives.
    """

    def __init__(self, population, incubation_time, recovery_time, data):
        compartments = ("S", "E", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(incubation_time, float)
        assert incubation_time > 1
        self.incubation_time = incubation_time

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data

    series = ("S2E", "E2I", "I2R", "obs")
    full_mass = [("R0", "rho")]

    def global_model(self):
        tau_e = self.incubation_time
        tau_i = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Beta(2, 2))
        return R0, tau_e, tau_i, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "E": 0, "I": 1}

    def transition_fwd(self, params, state, t):
        R0, tau_e, tau_i, rho = params

        # Sample flows between compartments.
        S2E = pyro.sample("S2E_{}".format(t),
                          infection_dist(individual_rate=R0 / tau_i,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population))
        E2I = pyro.sample("E2I_{}".format(t),
                          dist.Binomial(state["E"], 1 / tau_e))
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], 1 / tau_i))

        # Update compartments with flows.
        state["S"] = state["S"] - S2E
        state["E"] = state["E"] + S2E - E2I
        state["I"] = state["I"] + E2I - I2R

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2E, rho),
                    obs=self.data[t] if t < self.duration else None)

    def transition_bwd(self, params, prev, curr, t):
        R0, tau_e, tau_i, rho = params

        # Reverse the flow computation.
        S2E = prev["S"] - curr["S"]
        E2I = prev["E"] - curr["E"] + S2E
        I2R = prev["I"] - curr["I"] + E2I

        # Condition on flows between compartments.
        pyro.sample("S2E_{}".format(t),
                    infection_dist(individual_rate=R0 / tau_i,
                                   num_susceptible=prev["S"],
                                   num_infectious=prev["I"],
                                   population=self.population),
                    obs=S2E)
        pyro.sample("E2I_{}".format(t),
                    dist.ExtendedBinomial(prev["E"], 1 / tau_e),
                    obs=E2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(prev["I"], 1 / tau_i),
                    obs=I2R)

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2E, rho),
                    obs=self.data[t])


class OverdispersedSEIRModel(CompartmentalModel):
    r"""
    Overdispersed Susceptible-Exposed-Infected-Recovered model.

    To customize this model we recommend forking and editing this class.

    This is a stochastic discrete-time discrete-state model with four
    compartments: "S" for susceptible, "E" for exposed, "I" for infected,
    and "R" for recovered individuals (the recovered individuals are
    implicit: ``R = population - S - E - I``) with transitions
    ``S -> E -> I -> R``.

    This model accounts for superspreading (overdispersed individual
    reproductive number) by assuming each infected individual infects
    BetaBinomial-many susceptible individuals, where the BetaBinomial
    distribution acts as an overdispersed Binomial distribution, adapting the
    more standard NegativeBinomial distribution that acts as an overdispersed
    Poisson distribution [1,2] to the setting of finite populations. To
    preserve Markov structure, we follow [2] and assume all infections by a
    single individual occur on the single time step where that individual makes
    an ``I -> R`` transition. That is, whereas the :class:`SimpleSEIRModel`
    assumes infected individuals infect `Binomial(S,R/tau)`-many susceptible
    individuals during each infected time step (over `tau`-many steps on
    average), this model assumes they infect `BetaBinomial(k,...,S)`-many
    susceptible individuals but only on the final time step before recovering.

    This model also adds an optional likelihood for observed phylogenetic data
    in the form of coalescent times. These are provided as a pair
    ``(leaf_times, coal_times)`` of times at which genomes are sequenced and
    lineages coalesce, respectively. We incorporate this data using the
    :class:`~pyro.distributions.CoalescentRateLikelihood` with base coalescence
    rate computed from the ``S`` and ``I`` populations. This likelihood is
    independent across time and preserves the Markov propert needed for
    inference.

    **References**

    [1] J. O. Lloyd-Smith, S. J. Schreiber, P. E. Kopp, W. M. Getz (2005)
        "Superspreading and the effect of individual variation on disease
        emergence"
        https://www.nature.com/articles/nature04153.pdf
    [2] Lucy M. Li, Nicholas C. Grassly, Christophe Fraser (2017)
        "Quantifying Transmission Heterogeneity Using Both Pathogen Phylogenies
        and Incidence Time Series"
        https://academic.oup.com/mbe/article/34/11/2982/3952784

    :param int population: Total ``population = S + E + I + R``.
    :param float incubation_time: Mean incubation time (duration in state
        ``E``). Must be greater than 1.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param iterable data: Time series of new observed infections. Each time
        step is Binomial distributed between 0 and the number of ``S -> E``
        transitions. This allows false negative but no false positives.
    """

    def __init__(self, population, incubation_time, recovery_time, data, *,
                 leaf_times=None, coal_times=None):
        compartments = ("S", "E", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(incubation_time, float)
        assert incubation_time > 1
        self.incubation_time = incubation_time

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data

        assert (leaf_times is None) == (coal_times is None)
        if leaf_times is None:
            self.coal_likelihood = None
        else:
            self.coal_likelihood = dist.CoalescentRateLikelihood(
                leaf_times, coal_times, duration)

    series = ("S2E", "E2I", "I2R", "obs")
    full_mass = [("R0", "rho", "k")]

    def global_model(self):
        tau_e = self.incubation_time
        tau_i = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        k = pyro.sample("k", dist.Exponential(1.))
        rho = pyro.sample("rho", dist.Beta(2, 2))
        return R0, k, tau_e, tau_i, rho

    def initialize(self, params):
        # Start with a single exposure.
        return {"S": self.population - 1, "E": 0, "I": 1}

    def transition_fwd(self, params, state, t):
        R0, k, tau_e, tau_i, rho = params

        # Sample flows between compartments.
        E2I = pyro.sample("E2I_{}".format(t),
                          dist.Binomial(state["E"], 1 / tau_e))
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], 1 / tau_i))
        S2E = pyro.sample("S2E_{}".format(t),
                          infection_dist(individual_rate=R0,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population,
                                         concentration=k))

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2E, rho),
                    obs=self.data[t] if t < self.duration else None)
        if self.coal_likelihood is not None and t < self.duration:
            R = R0 * state["S"] / self.population
            coal_rate = R * (1. + 1. / k) / (tau_i * state["I"] + 1e-8)
            pyro.factor("coalescent_{}".format(t),
                        self.coal_likelihood(coal_rate, t))

        # Update compartements with flows.
        state["S"] = state["S"] - S2E
        state["E"] = state["E"] + S2E - E2I
        state["I"] = state["I"] + E2I - I2R

    def transition_bwd(self, params, prev, curr, t):
        R0, k, tau_e, tau_i, rho = params

        # Reverse the flow computation.
        S2E = prev["S"] - curr["S"]
        E2I = prev["E"] - curr["E"] + S2E
        I2R = prev["I"] - curr["I"] + E2I

        # Condition on flows between compartments.
        pyro.sample("S2E_{}".format(t),
                    infection_dist(individual_rate=R0,
                                   num_susceptible=prev["S"],
                                   num_infectious=prev["I"],
                                   population=self.population,
                                   concentration=k),
                    obs=S2E)
        pyro.sample("E2I_{}".format(t),
                    dist.ExtendedBinomial(prev["E"], 1 / tau_e),
                    obs=E2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(prev["I"], 1 / tau_i),
                    obs=I2R)

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2E, rho),
                    obs=self.data[t])
        if self.coal_likelihood is not None:
            R = R0 * prev["S"] / self.population
            coal_rate = R * (1. + 1. / k) / (tau_i * prev["I"] + 1e-8)
            pyro.factor("coalescent_{}".format(t),
                        self.coal_likelihood(coal_rate, t))
