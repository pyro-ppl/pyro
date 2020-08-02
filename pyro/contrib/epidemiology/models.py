# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import re

import torch
from torch.nn.functional import pad

import pyro
import pyro.distributions as dist

from .compartmental import CompartmentalModel
from .distributions import binomial_dist, infection_dist


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
    :param iterable data: Time series of new observed infections. Each time
        step is Binomial distributed between 0 and the number of ``S -> I``
        transitions. This allows false negative but no false positives.
    """

    def __init__(self, population, recovery_time, data):
        compartments = ("S", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Beta(10, 10))
        return R0, tau, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "I": 1}

    def transition(self, params, state, t):
        R0, tau, rho = params

        # Sample flows between compartments.
        S2I = pyro.sample("S2I_{}".format(t),
                          infection_dist(individual_rate=R0 / tau,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population))
        I2R = pyro.sample("I2R_{}".format(t),
                          binomial_dist(state["I"], 1 / tau))

        # Update compartments with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R

        # Condition on observations.
        t_is_observed = isinstance(t, slice) or t < self.duration
        pyro.sample("obs_{}".format(t),
                    binomial_dist(S2I, rho),
                    obs=self.data[t] if t_is_observed else None)


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

    def global_model(self):
        tau_e = self.incubation_time
        tau_i = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Beta(10, 10))
        return R0, tau_e, tau_i, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "E": 0, "I": 1}

    def transition(self, params, state, t):
        R0, tau_e, tau_i, rho = params

        # Sample flows between compartments.
        S2E = pyro.sample("S2E_{}".format(t),
                          infection_dist(individual_rate=R0 / tau_i,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population))
        E2I = pyro.sample("E2I_{}".format(t),
                          binomial_dist(state["E"], 1 / tau_e))
        I2R = pyro.sample("I2R_{}".format(t),
                          binomial_dist(state["I"], 1 / tau_i))

        # Update compartments with flows.
        state["S"] = state["S"] - S2E
        state["E"] = state["E"] + S2E - E2I
        state["I"] = state["I"] + E2I - I2R

        # Condition on observations.
        t_is_observed = isinstance(t, slice) or t < self.duration
        pyro.sample("obs_{}".format(t),
                    binomial_dist(S2E, rho),
                    obs=self.data[t] if t_is_observed else None)


class SimpleSEIRDModel(CompartmentalModel):
    """
    Susceptible-Exposed-Infected-Recovered-Dead model.

    To customize this model we recommend forking and editing this class.

    This is a stochastic discrete-time discrete-state model with four
    compartments: "S" for susceptible, "E" for exposed, "I" for infected, "D"
    for deceased individuals, and "R" for recovered individuals (the recovered
    individuals are implicit: ``R = population - S - E - I - D``) with
    transitions ``S -> E -> I -> R`` and ``I -> D``.

    Because the transitions are not simple linear succession, this model
    implements a custom :meth:`compute_flows()` method.

    :param int population: Total ``population = S + E + I + R + D``.
    :param float incubation_time: Mean incubation time (duration in state
        ``E``). Must be greater than 1.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param float mortality_rate: Portion of infections resulting in death.
        Must be in the open interval ``(0, 1)``.
    :param iterable data: Time series of new observed infections. Each time
        step is Binomial distributed between 0 and the number of ``S -> E``
        transitions. This allows false negative but no false positives.
    """

    def __init__(self, population, incubation_time, recovery_time,
                 mortality_rate, data):
        compartments = ("S", "E", "I", "D")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(incubation_time, float)
        assert incubation_time > 1
        self.incubation_time = incubation_time

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        assert isinstance(mortality_rate, float)
        assert 0 < mortality_rate < 1
        self.mortality_rate = mortality_rate

        self.data = data

    def global_model(self):
        tau_e = self.incubation_time
        tau_i = self.recovery_time
        mu = self.mortality_rate
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Beta(10, 10))
        return R0, tau_e, tau_i, mu, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "E": 0, "I": 1, "D": 0}

    def transition(self, params, state, t):
        R0, tau_e, tau_i, mu, rho = params

        # Sample flows between compartments.
        S2E = pyro.sample("S2E_{}".format(t),
                          infection_dist(individual_rate=R0 / tau_i,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population))
        E2I = pyro.sample("E2I_{}".format(t),
                          binomial_dist(state["E"], 1 / tau_e))
        # Of the 1/tau_i expected recoveries-or-deaths, a portion mu die and
        # the remaining recover. Alternatively we could model this with a
        # Multinomial distribution I2_ and extract the two components I2D and
        # I2R, however the Multinomial distribution does not currently
        # implement overdispersion or moment matching.
        I2D = pyro.sample("I2D_{}".format(t),
                          binomial_dist(state["I"], mu / tau_i))
        I2R = pyro.sample("I2R_{}".format(t),
                          binomial_dist(state["I"] - I2D, 1 / tau_i))

        # Update compartments with flows.
        state["S"] = state["S"] - S2E
        state["E"] = state["E"] + S2E - E2I
        state["I"] = state["I"] + E2I - I2R - I2D
        state["D"] = state["D"] + I2D

        # Condition on observations.
        t_is_observed = isinstance(t, slice) or t < self.duration
        pyro.sample("obs_{}".format(t),
                    binomial_dist(S2E, rho),
                    obs=self.data[t] if t_is_observed else None)

    def compute_flows(self, prev, curr, t):
        S2E = prev["S"] - curr["S"]  # S can only go to E.
        I2D = curr["D"] - prev["D"]  # D can only have come from I.
        # We deduce the remaining flows by conservation of mass:
        #   curr - prev = inflows - outflows
        E2I = prev["E"] - curr["E"] + S2E
        I2R = prev["I"] - curr["I"] + E2I - I2D
        return {
            "S2E_{}".format(t): S2E,
            "E2I_{}".format(t): E2I,
            "I2D_{}".format(t): I2D,
            "I2R_{}".format(t): I2R,
        }


class OverdispersedSIRModel(CompartmentalModel):
    """
    Generalizes :class:`SimpleSIRModel` with overdispersed distributions.

    To customize this model we recommend forking and editing this class.

    This adds a single global overdispersion parameter controlling
    overdispersion of the transition and observation distributions. See
    :func:`~pyro.contrib.epidemiology.distributions.binomial_dist` and
    :func:`~pyro.contrib.epidemiology.distributions.beta_binomial_dist` for
    distributional details. For prior work incorporating overdispersed
    distributions see [1,2,3,4].

    **References:**

    [1] D. Champredon, M. Li, B. Bolker. J. Dushoff (2018)
        "Two approaches to forecast Ebola synthetic epidemics"
        https://www.sciencedirect.com/science/article/pii/S1755436517300233
    [2] Carrie Reed et al. (2015)
        "Estimating Influenza Disease Burden from Population-Based Surveillance
        Data in the United States"
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349859/
    [3] A. Leonard, D. Weissman, B. Greenbaum, E. Ghedin, K. Koelle (2017)
        "Transmission Bottleneck Size Estimation from Pathogen Deep-Sequencing
        Data, with an Application to Human Influenza A Virus"
        https://jvi.asm.org/content/jvi/91/14/e00171-17.full.pdf
    [4] A. Miller, N. Foti, J. Lewnard, N. Jewell, C. Guestrin, E. Fox (2020)
        "Mobility trends provide a leading indicator of changes in
        SARS-CoV-2 transmission"
        https://www.medrxiv.org/content/medrxiv/early/2020/05/11/2020.05.07.20094441.full.pdf

    :param int population: Total ``population = S + I + R``.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param iterable data: Time series of new observed infections. Each time
        step is Binomial distributed between 0 and the number of ``S -> I``
        transitions. This allows false negative but no false positives.
    """

    def __init__(self, population, recovery_time, data):
        compartments = ("S", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Beta(10, 10))
        od = pyro.sample("od", dist.Beta(2, 6))
        return R0, tau, rho, od

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "I": 1}

    def transition(self, params, state, t):
        R0, tau, rho, od = params

        # Sample flows between compartments.
        S2I = pyro.sample("S2I_{}".format(t),
                          infection_dist(individual_rate=R0 / tau,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population,
                                         overdispersion=od))
        I2R = pyro.sample("I2R_{}".format(t),
                          binomial_dist(state["I"], 1 / tau,
                                        overdispersion=od))

        # Update compartments with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R

        # Condition on observations.
        t_is_observed = isinstance(t, slice) or t < self.duration
        pyro.sample("obs_{}".format(t),
                    binomial_dist(S2I, rho, overdispersion=od),
                    obs=self.data[t] if t_is_observed else None)


class OverdispersedSEIRModel(CompartmentalModel):
    """
    Generalizes :class:`SimpleSEIRModel` with overdispersed distributions.

    To customize this model we recommend forking and editing this class.

    This adds a single global overdispersion parameter controlling
    overdispersion of the transition and observation distributions. See
    :func:`~pyro.contrib.epidemiology.distributions.binomial_dist` and
    :func:`~pyro.contrib.epidemiology.distributions.beta_binomial_dist` for
    distributional details. For prior work incorporating overdispersed
    distributions see [1,2,3,4].

    **References:**

    [1] D. Champredon, M. Li, B. Bolker. J. Dushoff (2018)
        "Two approaches to forecast Ebola synthetic epidemics"
        https://www.sciencedirect.com/science/article/pii/S1755436517300233
    [2] Carrie Reed et al. (2015)
        "Estimating Influenza Disease Burden from Population-Based Surveillance
        Data in the United States"
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4349859/
    [3] A. Leonard, D. Weissman, B. Greenbaum, E. Ghedin, K. Koelle (2017)
        "Transmission Bottleneck Size Estimation from Pathogen Deep-Sequencing
        Data, with an Application to Human Influenza A Virus"
        https://jvi.asm.org/content/jvi/91/14/e00171-17.full.pdf
    [4] A. Miller, N. Foti, J. Lewnard, N. Jewell, C. Guestrin, E. Fox (2020)
        "Mobility trends provide a leading indicator of changes in
        SARS-CoV-2 transmission"
        https://www.medrxiv.org/content/medrxiv/early/2020/05/11/2020.05.07.20094441.full.pdf

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

    def global_model(self):
        tau_e = self.incubation_time
        tau_i = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Beta(10, 10))
        od = pyro.sample("od", dist.Beta(2, 6))
        return R0, tau_e, tau_i, rho, od

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "E": 0, "I": 1}

    def transition(self, params, state, t):
        R0, tau_e, tau_i, rho, od = params

        # Sample flows between compartments.
        S2E = pyro.sample("S2E_{}".format(t),
                          infection_dist(individual_rate=R0 / tau_i,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population,
                                         overdispersion=od))
        E2I = pyro.sample("E2I_{}".format(t),
                          binomial_dist(state["E"], 1 / tau_e, overdispersion=od))
        I2R = pyro.sample("I2R_{}".format(t),
                          binomial_dist(state["I"], 1 / tau_i, overdispersion=od))

        # Update compartments with flows.
        state["S"] = state["S"] - S2E
        state["E"] = state["E"] + S2E - E2I
        state["I"] = state["I"] + E2I - I2R

        # Condition on observations.
        t_is_observed = isinstance(t, slice) or t < self.duration
        pyro.sample("obs_{}".format(t),
                    binomial_dist(S2E, rho, overdispersion=od),
                    obs=self.data[t] if t_is_observed else None)


class SuperspreadingSIRModel(CompartmentalModel):
    """
    Generalizes :class:`SimpleSIRModel` by adding superspreading effects.

    To customize this model we recommend forking and editing this class.

    This model accounts for superspreading (overdispersed individual
    reproductive number) by assuming each infected individual infects
    BetaBinomial-many susceptible individuals, where the BetaBinomial
    distribution acts as an overdispersed Binomial distribution, adapting the
    more standard NegativeBinomial distribution that acts as an overdispersed
    Poisson distribution [1,2] to the setting of finite populations. To
    preserve Markov structure, we follow [2] and assume all infections by a
    single individual occur on the single time step where that individual makes
    an ``I -> R`` transition. That is, whereas the :class:`SimpleSIRModel`
    assumes infected individuals infect `Binomial(S,R/tau)`-many susceptible
    individuals during each infected time step (over `tau`-many steps on
    average), this model assumes they infect `BetaBinomial(k,...,S)`-many
    susceptible individuals but only on the final time step before recovering.

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
    :param iterable data: Time series of new observed infections. Each time
        step is Binomial distributed between 0 and the number of ``S -> I``
        transitions. This allows false negative but no false positives.
    """

    def __init__(self, population, recovery_time, data):
        compartments = ("S", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        k = pyro.sample("k", dist.Exponential(1.))
        rho = pyro.sample("rho", dist.Beta(10, 10))
        return R0, k, tau, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "I": 1}

    def transition(self, params, state, t):
        R0, k, tau, rho = params

        # Sample flows between compartments.
        I2R = pyro.sample("I2R_{}".format(t),
                          binomial_dist(state["I"], 1 / tau))
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
        t_is_observed = isinstance(t, slice) or t < self.duration
        pyro.sample("obs_{}".format(t),
                    binomial_dist(S2I, rho),
                    obs=self.data[t] if t_is_observed else None)


class SuperspreadingSEIRModel(CompartmentalModel):
    r"""
    Generalizes :class:`SimpleSEIRModel` by adding superspreading effects.

    To customize this model we recommend forking and editing this class.

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

    def global_model(self):
        tau_e = self.incubation_time
        tau_i = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        k = pyro.sample("k", dist.Exponential(1.))
        rho = pyro.sample("rho", dist.Beta(10, 10))
        return R0, k, tau_e, tau_i, rho

    def initialize(self, params):
        # Start with a single exposure.
        return {"S": self.population - 1, "E": 0, "I": 1}

    def transition(self, params, state, t):
        R0, k, tau_e, tau_i, rho = params

        # Sample flows between compartments.
        E2I = pyro.sample("E2I_{}".format(t),
                          binomial_dist(state["E"], 1 / tau_e))
        I2R = pyro.sample("I2R_{}".format(t),
                          binomial_dist(state["I"], 1 / tau_i))
        S2E = pyro.sample("S2E_{}".format(t),
                          infection_dist(individual_rate=R0,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population,
                                         concentration=k))

        # Condition on observations.
        t_is_observed = isinstance(t, slice) or t < self.duration
        pyro.sample("obs_{}".format(t),
                    binomial_dist(S2E, rho),
                    obs=self.data[t] if t_is_observed else None)
        if self.coal_likelihood is not None:
            R = R0 * state["S"] / self.population
            coal_rate = R * (1. + 1. / k) / (tau_i * state["I"] + 1e-8)
            pyro.factor("coalescent_{}".format(t),
                        self.coal_likelihood(coal_rate, t)
                        if t_is_observed else torch.tensor(0.))

        # Update compartements with flows.
        state["S"] = state["S"] - S2E
        state["E"] = state["E"] + S2E - E2I
        state["I"] = state["I"] + E2I - I2R


class HeterogeneousSIRModel(CompartmentalModel):
    """
    Generalizes :class:`SimpleSIRModel` by allowing ``Rt`` and ``rho`` to vary
    in time.

    To customize this model we recommend forking and editing this class.

    In this model, the response rate ``rho`` is piecewise constant with unknown
    value over three pieces. The reproductive number ``Rt`` is a product of a
    constant ``R0`` with a factor ``beta`` that drifts via Brownian motion in
    log space. Both ``rho`` and ``Rt`` are available as time series.

    :param int population: Total ``population = S + I + R``.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param iterable data: Time series of new observed infections. Each time
        step is Binomial distributed between 0 and the number of ``S -> I``
        transitions. This allows false negative but no false positives.
    """

    def __init__(self, population, recovery_time, data):
        compartments = ("S", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))

        # Let's consider a piecewise constant response rate, say low rate for
        # two weeks, then intermediate rate as testing capacity increases, and
        # finally high rate for a few months (as far into the future as we'd
        # like to forecast). We don't know exactly what the rates are, but we
        # can specify increasingly informative priors.
        rho0 = pyro.sample("rho0", dist.Beta(2, 4))
        rho1 = pyro.sample("rho1", dist.Beta(4, 4))
        rho2 = pyro.sample("rho2", dist.Beta(8, 4))
        # Later .transition() will index into this time series as rho[..., t].
        rho = torch.cat([rho0.unsqueeze(-1).expand(rho0.shape + (14,)),
                         rho1.unsqueeze(-1).expand(rho1.shape + (7,)),
                         rho2.unsqueeze(-1).expand(rho2.shape + (60,))], dim=-1)
        # We can also save the time series for output in self.samples.
        pyro.deterministic("rho", rho, event_dim=1)

        return R0, tau, rho

    def initialize(self, params):
        R0, tau, rho = params
        # Start with a single infection.
        # We also store the initial beta value in the state dict.
        return {"S": self.population - 1, "I": 1, "beta": torch.tensor(1.)}

    def transition(self, params, state, t):
        R0, tau, rho = params

        # Sample heterogeneous variables.
        # This assumes beta slowly drifts via Brownian motion in log space.
        beta = pyro.sample("beta_{}".format(t),
                           dist.LogNormal(state["beta"].log(), 0.1))
        Rt = pyro.deterministic("Rt_{}".format(t), R0 * beta)

        # Sample flows between compartments.
        S2I = pyro.sample("S2I_{}".format(t),
                          infection_dist(individual_rate=Rt / tau,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population))
        I2R = pyro.sample("I2R_{}".format(t),
                          binomial_dist(state["I"], 1 / tau))

        # Update compartments and heterogeneous variables.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R
        state["beta"] = beta  # We store the latest beta value in the state dict.

        # Condition on observations.
        # Note that, since rho may be batched over particles or samples, we
        # need to index it via rho[..., t] rather than a simple rho[t].
        t_is_observed = isinstance(t, slice) or t < self.duration
        pyro.sample("obs_{}".format(t),
                    binomial_dist(S2I, rho[..., t]),
                    obs=self.data[t] if t_is_observed else None)


class SparseSIRModel(CompartmentalModel):
    """
    Generalizes :class:`SimpleSIRModel` to allow sparsely observed infections.

    To customize this model we recommend forking and editing this class.

    This model allows observations of **cumulative** infections at uneven time
    intervals. To preserve Markov structure (and hence tractable inference)
    this model adds an auxiliary compartment ``O`` denoting the fully-observed
    cumulative number of observations at each time point. At observed times
    (when ``mask[t] == True``) ``O`` must exactly match the provided data;
    between observed times ``O`` stochastically imputes the provided data.

    This model demonstrates how to implement a custom :meth:`compute_flows`
    method. A custom method is needed in this model because inhabitants of the
    ``S`` compartment can transition to both the ``I`` and ``O`` compartments,
    allowing duplication.

    :param int population: Total ``population = S + I + R``.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param iterable data: Time series of **cumulative** observed infections.
        Whenever ``mask[t] == True``, ``data[t]`` corresponds to an
        observation; otherwise ``data[t]`` can be arbitrary, e.g. NAN.
    :param iterable mask: Boolean time series denoting whether an observation
        is made at each time step. Should satisfy ``len(mask) == len(data)``.
    """

    def __init__(self, population, recovery_time, data, mask):
        assert len(data) == len(mask)
        duration = len(data)
        compartments = ("S", "I", "O")  # O is auxiliary, R is implicit.
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data
        self.mask = mask

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Beta(10, 10))
        return R0, tau, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "I": 1, "O": 0}

    def transition(self, params, state, t):
        R0, tau, rho = params

        # Sample flows between compartments.
        S2I = pyro.sample("S2I_{}".format(t),
                          infection_dist(individual_rate=R0 / tau,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"],
                                         population=self.population))
        I2R = pyro.sample("I2R_{}".format(t),
                          binomial_dist(state["I"], 1 / tau))
        S2O = pyro.sample("S2O_{}".format(t),
                          binomial_dist(S2I, rho))

        # Update compartments with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R
        state["O"] = state["O"] + S2O

        # Condition on cumulative observations.
        t_is_observed = isinstance(t, slice) or t < self.duration
        mask_t = self.mask[t] if t_is_observed else False
        data_t = self.data[t] if t_is_observed else None
        pyro.sample("obs_{}".format(t),
                    # FIXME Delta is incompatible with relaxed inference.
                    dist.Delta(state["O"]).mask(mask_t),
                    obs=data_t)

    def compute_flows(self, prev, curr, t):
        # Reverse the flow computation.
        S2I = prev["S"] - curr["S"]
        I2R = prev["I"] - curr["I"] + S2I
        S2O = curr["O"] - prev["O"]
        return {
            "S2I_{}".format(t): S2I,
            "I2R_{}".format(t): I2R,
            "S2O_{}".format(t): S2O,
        }


class UnknownStartSIRModel(CompartmentalModel):
    """
    Generalizes :class:`SimpleSIRModel` by allowing unknown date of first
    infection.

    To customize this model we recommend forking and editing this class.

    This model demonstrates:

    1.  How to incorporate spontaneous infections from external sources;
    2.  How to incorporate time-varying piecewise ``rho`` by supporting
        forecasting in :meth:`transition`.
    3.  How to override the :meth:`predict` method to compute extra
        statistics.

    :param int population: Total ``population = S + I + R``.
    :param float recovery_time: Mean recovery time (duration in state
        ``I``). Must be greater than 1.
    :param int pre_obs_window: Number of time steps before beginning ``data``
        where the initial infection may have occurred. Must be positive.
    :param iterable data: Time series of new observed infections. Each time
        step is Binomial distributed between 0 and the number of ``S -> I``
        transitions. This allows false negative but no false positives.
    """

    def __init__(self, population, recovery_time, pre_obs_window, data):
        compartments = ("S", "I")  # R is implicit.
        duration = pre_obs_window + len(data)
        super().__init__(compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        assert isinstance(pre_obs_window, int) and pre_obs_window > 0
        self.pre_obs_window = pre_obs_window
        self.post_obs_window = len(data)

        # We set a small time-constant external infecton rate such that on
        # average there is a single external infection during the
        # pre_obs_window. This allows unknown time of initial infection
        # without introducing long-range coupling across time.
        self.external_rate = 1 / pre_obs_window

        # Prepend data with zeros.
        if isinstance(data, list):
            data = [0.] * self.pre_obs_window + data
        else:
            data = pad(data, (self.pre_obs_window, 0), value=0.)
        self.data = data

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))

        # Assume two different response rates: rho0 before any observations
        # were made (in pre_obs_window), followed by a higher response rate rho1
        # after observations were made (in post_obs_window).
        rho0 = pyro.sample("rho0", dist.Beta(10, 10))
        rho1 = pyro.sample("rho1", dist.Beta(10, 10))
        # Whereas each of rho0,rho1 are scalars (possibly batched over samples),
        # we construct a time series rho with an extra time dim on the right.
        rho = torch.cat([
            rho0.unsqueeze(-1).expand(rho0.shape + (self.pre_obs_window,)),
            rho1.unsqueeze(-1).expand(rho1.shape + (self.post_obs_window,)),
        ], dim=-1)

        # Model external infections as an infectious pseudo-individual added
        # to num_infectious when sampling S2I below.
        X = self.external_rate * tau / R0

        return R0, X, tau, rho

    def initialize(self, params):
        # Start with no internal infections.
        return {"S": self.population, "I": 0}

    def transition(self, params, state, t):
        R0, X, tau, rho = params

        # Sample flows between compartments.
        S2I = pyro.sample("S2I_{}".format(t),
                          infection_dist(individual_rate=R0 / tau,
                                         num_susceptible=state["S"],
                                         num_infectious=state["I"] + X,
                                         population=self.population))
        I2R = pyro.sample("I2R_{}".format(t),
                          binomial_dist(state["I"], 1 / tau))

        # Update compartments with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R

        # In .transition() t will always be an integer but may lie outside
        # of [0,self.duration) when forecasting.
        t_is_observed = isinstance(t, slice) or t < self.duration
        rho_t = rho[..., t] if t_is_observed else rho[..., -1]
        data_t = self.data[t] if t_is_observed else None

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    binomial_dist(S2I, rho_t),
                    obs=data_t)

    def predict(self, forecast=0):
        """
        Augments
        :meth:`~pyro.contrib.epidemiology.compartmental.Compartmental.predict`
        with samples of ``first_infection`` i.e. the first time index at which
        the infection ``I`` becomes nonzero. Note this is measured from the
        beginning of ``pre_obs_window``, not the beginning of data.

        :param int forecast: The number of time steps to forecast forward.
        :returns: A dictionary mapping sample site name (or compartment name)
            to a tensor whose first dimension corresponds to sample batching.
        :rtype: dict
        """
        samples = super().predict(forecast)

        # Extract the time index of the first infection (samples["I"] > 0)
        # for each sample trajectory in the samples["I"] tensor.
        samples["first_infection"] = samples["I"].cumsum(-1).eq(0).sum(-1)

        return samples


class RegionalSIRModel(CompartmentalModel):
    r"""
    Generalizes :class:`SimpleSIRModel` to simultaneously model multiple
    regions with weak coupling across regions.

    To customize this model we recommend forking and editing this class.

    Regions are coupled by a ``coupling`` matrix with entries in ``[0,1]``.
    The all ones matrix is equivalent to a single region. The identity matrix
    is equivalent to a set of independent regions. This need not be symmetric,
    but symmetric matrices are probably more physically plausible. The expected
    number of new infections each time step ``S2I`` is Binomial distributed
    with mean::

        E[S2I] = S (1 - (1 - R0 / (population @ coupling)) ** (I @ coupling))
               ≈ R0 S (I @ coupling) / (population @ coupling)  # for small I

    Thus in a nearly entirely susceptible population, a single infected
    individual infects approximately ``R0`` new individuals on average,
    independent of ``coupling``.

    This model demonstrates:

    1.  How to create a regional model with a ``population`` vector.
    2.  How to model both homogeneous parameters (here ``R0``) and
        heterogeneous parameters with hierarchical structure (here ``rho``)
        using ``self.region_plate``.
    3.  How to approximately couple regions in :meth:`transition` using
        ``state["I_approx"]``.

    :param torch.Tensor population: Tensor of per-region populations, defining
        ``population = S + I + R``.
    :param torch.Tensor coupling: Pairwise coupling matrix. Entries should be
        in ``[0,1]``.
    :param float recovery_time: Mean recovery time (duration in state ``I``).
        Must be greater than 1.
    :param iterable data: Time x Region sized tensor of new observed
        infections. Each time step is vector of Binomials distributed between
        0 and the number of ``S -> I`` transitions. This allows false negative
        but no false positives.
    """

    def __init__(self, population, coupling, recovery_time, data):
        duration = len(data)
        num_regions, = population.shape
        assert coupling.shape == (num_regions, num_regions)
        assert (0 <= coupling).all()
        assert (coupling <= 1).all()
        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        if isinstance(data, torch.Tensor):
            # Data tensors should be oriented as (time, region).
            assert data.shape == (duration, num_regions)
        compartments = ("S", "I")  # R is implicit.

        # We create a regional model by passing a vector of populations.
        super().__init__(compartments, duration, population, approximate=("I",))

        self.coupling = coupling
        self.recovery_time = recovery_time
        self.data = data

    def global_model(self):
        # Assume recovery time is a known constant.
        tau = self.recovery_time

        # Assume reproductive number is unknown but homogeneous.
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))

        # Assume response rate is heterogeneous and model it with a
        # hierarchical Gamma-Beta prior.
        rho_c1 = pyro.sample("rho_c1", dist.Gamma(10, 1))
        rho_c0 = pyro.sample("rho_c0", dist.Gamma(10, 1))
        with self.region_plate:
            rho = pyro.sample("rho", dist.Beta(rho_c1, rho_c0))

        return R0, tau, rho

    def initialize(self, params):
        # Start with a single infection in region 0.
        I = torch.zeros_like(self.population)
        I[0] += 1
        S = self.population - I
        return {"S": S, "I": I}

    def transition(self, params, state, t):
        R0, tau, rho = params

        # Account for infections from all regions. This uses approximate (point
        # estimate) counts I_approx for infection from other regions, but uses
        # the exact (enumerated) count I for infections from one's own region.
        I_coupled = state["I_approx"] @ self.coupling
        I_coupled = I_coupled + (state["I"] - state["I_approx"]) * self.coupling.diag()
        I_coupled = I_coupled.clamp(min=0)  # In case I_approx is negative.
        pop_coupled = self.population @ self.coupling

        with self.region_plate:
            # Sample flows between compartments.
            S2I = pyro.sample("S2I_{}".format(t),
                              infection_dist(individual_rate=R0 / tau,
                                             num_susceptible=state["S"],
                                             num_infectious=I_coupled,
                                             population=pop_coupled))
            I2R = pyro.sample("I2R_{}".format(t),
                              binomial_dist(state["I"], 1 / tau))

            # Update compartments with flows.
            state["S"] = state["S"] - S2I
            state["I"] = state["I"] + S2I - I2R

            # Condition on observations.
            t_is_observed = isinstance(t, slice) or t < self.duration
            pyro.sample("obs_{}".format(t),
                        binomial_dist(S2I, rho),
                        obs=self.data[t] if t_is_observed else None)


class HeterogeneousRegionalSIRModel(CompartmentalModel):
    """
    Generalizes :class:`RegionalSIRModel` by allowing ``Rt`` and ``rho`` to vary
    in time.

    To customize this model we recommend forking and editing this class.

    In this model, the response rate ``rho`` varies across time and region,
    whereas the reproductive number ``Rt`` varies in time but is shared among
    regions. Both parameters drift according to transformed Brownian motion
    with learned drift rate.

    This model demonstrates how to model hierarchical latent time series,
    other than compartmental variables.

    :param torch.Tensor population: Tensor of per-region populations, defining
        ``population = S + I + R``.
    :param torch.Tensor coupling: Pairwise coupling matrix. Entries should be
        in ``[0,1]``.
    :param float recovery_time: Mean recovery time (duration in state ``I``).
        Must be greater than 1.
    :param iterable data: Time x Region sized tensor of new observed
        infections. Each time step is vector of Binomials distributed between
        0 and the number of ``S -> I`` transitions. This allows false negative
        but no false positives.
    """

    def __init__(self, population, coupling, recovery_time, data):
        duration = len(data)
        num_regions, = population.shape
        assert coupling.shape == (num_regions, num_regions)
        assert (0 <= coupling).all()
        assert (coupling <= 1).all()
        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        if isinstance(data, torch.Tensor):
            # Data tensors should be oriented as (time, region).
            assert data.shape == (duration, num_regions)
        compartments = ("S", "I")  # R is implicit.

        # We create a regional model by passing a vector of populations.
        super().__init__(compartments, duration, population, approximate=("I",))

        self.coupling = coupling
        self.recovery_time = recovery_time
        self.data = data

    def global_model(self):
        tau = self.recovery_time

        # Assume reproductive number is heterogeneous but shared among regions.
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        R_drift = pyro.sample("R_drift", dist.LogNormal(-3., 1.))

        # Assume response rate is heterogeneous in time and region.
        with self.region_plate:
            rho0 = pyro.sample("rho0", dist.Beta(4, 4))
        rho_drift = pyro.sample("rho_drift", dist.LogNormal(-3., 1.))

        return tau, R0, R_drift, rho0, rho_drift

    def initialize(self, params):
        # Start with a single infection in region 0.
        I = torch.zeros_like(self.population)
        I[0] += 1
        S = self.population - I
        return {"S": S, "I": I,
                "R_factor": torch.tensor(1.),
                "rho_shift": torch.tensor(0.)}

    def transition(self, params, state, t):
        tau, R0, R_drift, rho0, rho_drift = params

        # Account for infections from all regions. This uses approximate (point
        # estimate) counts I_approx for infection from other regions, but uses
        # the exact (enumerated) count I for infections from one's own region.
        I_coupled = state["I_approx"] @ self.coupling
        I_coupled = I_coupled + (state["I"] - state["I_approx"]) * self.coupling.diag()
        I_coupled = I_coupled.clamp(min=0)  # In case I_approx is negative.
        pop_coupled = self.population @ self.coupling

        # Sample region-global time-heterogeneous variables.
        R_factor = pyro.sample("R_factor_{}".format(t),
                               dist.LogNormal(state["R_factor"].log(), R_drift))
        Rt = pyro.deterministic("Rt_{}".format(t), R0 * R_factor)

        with self.region_plate:
            # Sample region-local time-heterogeneous variables.
            rho_shift = pyro.sample("rho_shift_{}".format(t),
                                    dist.Normal(state["rho_shift"], rho_drift))
            rho = pyro.deterministic("rho_{}".format(t),
                                     (rho0.log() - (-rho0).log1p() + rho_shift).sigmoid())

            # Sample flows between compartments.
            S2I = pyro.sample("S2I_{}".format(t),
                              infection_dist(individual_rate=Rt / tau,
                                             num_susceptible=state["S"],
                                             num_infectious=I_coupled,
                                             population=pop_coupled))
            I2R = pyro.sample("I2R_{}".format(t),
                              binomial_dist(state["I"], 1 / tau))

            # Update compartments and heterogeneous variables.
            state["S"] = state["S"] - S2I
            state["I"] = state["I"] + S2I - I2R
            state["R_factor"] = R_factor
            state["rho_shift"] = rho_shift

            # Condition on observations.
            t_is_observed = isinstance(t, slice) or t < self.duration
            pyro.sample("obs_{}".format(t),
                        binomial_dist(S2I, rho),
                        obs=self.data[t] if t_is_observed else None)


# Create sphinx documentation.
__all__ = []
for _name, _Model in list(locals().items()):
    if isinstance(_Model, type) and issubclass(_Model, CompartmentalModel):
        if _Model is not CompartmentalModel:
            __all__.append(_name)
__all__.sort(key=lambda name, vals=locals(): vals[name].__init__.__code__.co_firstlineno)
__doc__ = "\n\n".join([
    """
    {}
    ----------------------------------------------------------------
    .. autoclass:: pyro.contrib.epidemiology.models.{}
    """.format(re.sub("([A-Z][a-z]+)", r"\1 ", _name[:-5]), _name)
    for _name in __all__
])
