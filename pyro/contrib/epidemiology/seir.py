# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
from pyro.ops.tensor_utils import convolve

from .compartmental import CompartmentalModel


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
    :param iterable data: Time series of new observed infections.
    :param int num_quant_bins: The number of quantization bins to use.
        Note that computational cost is exponential in `num_quant_bins`.
        Defaults to 4.
    """

    def __init__(self, population, incubation_time, recovery_time, data, *,
                 num_quant_bins=4):
        compartments = ("S", "E", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population,
                         num_quant_bins=num_quant_bins)

        assert isinstance(incubation_time, float)
        assert incubation_time > 1
        self.incubation_time = incubation_time

        assert isinstance(recovery_time, float)
        assert recovery_time > 1
        self.recovery_time = recovery_time

        self.data = data

    series = ("S2E", "E2I", "I2R", "obs")
    full_mass = [("R0", "rho", "c")]

    def heuristic(self):
        T = len(self.data)
        # Start with a single exposure.
        S0 = self.population - 1
        # Assume 50% <= response rate <= 100%.
        I2R = self.data * min(2., (S0 / self.data.sum()).sqrt())
        # Assume recovery less than a month.
        recovery = torch.arange(30.).div(self.recovery_time).exp()
        recovery /= recovery.sum()
        E2I = convolve(I2R, recovery)
        E2I_cumsum = E2I[:-T].sum() + E2I[-T:].cumsum(-1)
        # Assume incubation takes less than a week.
        incubation = torch.arange(7.).div(self.incubation_time).exp()
        incubation /= incubation.sum()
        S2E = convolve(E2I, incubation)
        S2E_cumsum = S2E[:-T].sum() + S2E[-T:].cumsum(-1)
        # Accumulate.
        S_aux = S0 - S2E_cumsum
        E_aux = S2E_cumsum - E2I_cumsum
        I_aux = 1 + E2I_cumsum - I2R.cumsum(-1)

        return {
            "R0": torch.tensor(2.0),
            "rho": torch.tensor(0.5),
            "c": torch.tensor(1.0),
            "auxiliary": torch.stack([S_aux, E_aux, I_aux]).clamp(min=0.5),
        }

    def global_model(self):
        tau_e = self.incubation_time
        tau_i = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Uniform(0, 1))
        c = pyro.sample("c", dist.Exponential(1.))

        # Convert interpretable parameters to distribution parameters.
        rate_s = -R0 / self.population
        prob_e = 1 / tau_e
        prob_i = 1 / tau_i

        return c, rate_s, prob_e, prob_i, rho

    def initialize(self, params):
        # Start with a single exposure.
        return {"S": self.population - 1, "E": 1, "I": 0}

    def transition_fwd(self, params, state, t):
        c, rate_s, prob_e, prob_i, rho = params

        # Sample flows between compartments.
        E2I = pyro.sample("E2I_{}".format(t),
                          dist.Binomial(state["E"], prob_e))
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], prob_i))
        c0 = (c * (rate_s * I2R).exp()).clamp(min=1e-6)
        c1 = (c - c0).clamp(min=1e-6)
        S2E = pyro.sample("S2E_{}".format(t),
                          dist.BetaBinomial(c1, c0, state["S"]))

        # Update compartements with flows.
        state["S"] = state["S"] - S2E
        state["E"] = state["E"] + S2E - E2I
        state["I"] = state["I"] + E2I - I2R

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(E2I, rho),
                    obs=self.data[t] if t < self.duration else None)

    def transition_bwd(self, params, prev, curr, t):
        c, rate_s, prob_e, prob_i, rho = params

        # Reverse the flow computation.
        S2E = prev["S"] - curr["S"]
        E2I = prev["E"] - curr["E"] + S2E
        I2R = prev["I"] - curr["I"] + E2I

        # Condition on flows between compartments.
        c0 = (c * (rate_s * I2R).exp()).clamp(min=1e-6)
        c1 = (c - c0).clamp(min=1e-6)
        pyro.sample("S2E_{}".format(t),
                    dist.ExtendedBetaBinomial(c1, c0, prev["S"]),
                    obs=S2E)
        pyro.sample("E2I_{}".format(t),
                    dist.ExtendedBinomial(prev["E"], prob_e),
                    obs=E2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(prev["I"], prob_i),
                    obs=I2R)

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(E2I, rho),
                    obs=self.data[t])
