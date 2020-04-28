# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.nn.functional import pad

import pyro
import pyro.distributions as dist
from pyro.ops.tensor_utils import convolve

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

    def heuristic(self):
        # Start with a single infection.
        S0 = self.population - 1
        # Assume 50% <= response rate <= 100%.
        S2I = self.data * min(2., (S0 / self.data.sum()).sqrt())
        S_aux = S0 - S2I.cumsum(-1)
        # Account for the single initial infection.
        S2I[0] += 1
        # Assume infection lasts less than a month.
        recovery = torch.arange(30.).div(self.recovery_time).neg().exp()
        I_aux = convolve(S2I, recovery)[:len(self.data)]

        return {
            "R0": torch.tensor(2.0),
            "rho": torch.tensor(0.5),
            "auxiliary": torch.stack([S_aux, I_aux]).clamp(min=0.5),
        }

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

    At a given time step, let ``R0`` be the basic reproductive number, ``k`` be
    the dispersion, ``N`` be the total population, ``S`` be the number of
    susceptible individuals, and ``I2R`` be the number of individuals
    recovering at that time step. Then the reproductive number is

        Rt = R0 * S / N

    and the number of new infections ``S2I`` is distributed as::

        S2I ~ BetaBinomial(concentration1=k * I2R,
                           concentration2=k * S / Rt,
                           total_count=S)

            ≈ GammaPoisson(concentration=k * I2R,  # in the limit S → ∞
                           rate=K / Rt)

            = NegativeBinomial(R=Rt * I2R, k=k * I2R)

    **References**

    [1] J. O. Lloyd-Smith, S. J. Schreiber, P. E. Kopp, W. M. Getz
        "Superspreading and the effect of individual variation on disease
        emergence"
        Nature (2005)
        https://www.nature.com/articles/nature04153.pdf
    [2] Lucy M. Li, Nicholas C. Grassly, Christophe Fraser
        "Quantifying Transmission Heterogeneity Using Both Pathogen Phylogenies
        and Incidence Time Series"
        Molecular Biology and Evolution (2017)
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
    full_mass = [("R0", "rho")]

    def heuristic(self):
        T = len(self.data)
        # Start with a single exposure.
        S0 = self.population - 1
        # Assume 50% <= response rate <= 100%.
        I2R = self.data * min(2., (S0 / self.data.sum()).sqrt())
        # Assume recovery less than a month.
        recovery = torch.arange(30.).div(self.recovery_time).exp()
        recovery = pad(recovery, (0, 1), value=0)
        recovery /= recovery.sum()
        S2I = convolve(I2R, recovery)
        S2I_cumsum = S2I[:-T].sum() + S2I[-T:].cumsum(-1)
        # Accumulate.
        S_aux = S0 - S2I_cumsum
        I_aux = 1 + S2I_cumsum - I2R.cumsum(-1)

        return {
            "R0": torch.tensor(2.0),
            "rho": torch.tensor(0.5),
            "k": torch.tensor(1.0),
            "auxiliary": torch.stack([S_aux, I_aux]).clamp(min=0.5),
        }

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        k = pyro.sample("k", dist.Exponential(1.))
        rho = pyro.sample("rho", dist.Uniform(0, 1))

        # Convert interpretable parameters to distribution parameters.
        prob_i = 1 / tau

        return R0, k, prob_i, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "I": 1}

    def transition_fwd(self, params, state, t):
        R0, k, prob_i, rho = params

        # Sample flows between compartments.
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], prob_i))
        c1 = (k * I2R).clamp(min=1e-6)
        c0 = k / R0 * self.population
        S2I = pyro.sample("S2I_{}".format(t),
                          dist.BetaBinomial(c1, c0, state["S"]))

        # Update compartments with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t] if t < self.duration else None)

    def transition_bwd(self, params, prev, curr, t):
        R0, k, prob_i, rho = params

        # Reverse the flow computation.
        S2I = prev["S"] - curr["S"]
        I2R = prev["I"] - curr["I"] + S2I

        # Condition on flows between compartments.
        c1 = (k * I2R).clamp(min=1e-6)
        c0 = k / R0 * self.population
        pyro.sample("S2I_{}".format(t),
                    dist.ExtendedBetaBinomial(c1, c0, prev["S"]),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(prev["I"], prob_i),
                    obs=I2R)

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t])
