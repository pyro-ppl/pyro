# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
from pyro.ops.tensor_utils import convolve

from .compartmental import CompartmentalModel


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
    :param int data: Time series of new observed infections.
    :param int num_quant_bins: The number of quantization bins to use.
        Note that computational cost is exponential in `num_quant_bins`.
        Defaults to 4.
    """

    def __init__(self, population, recovery_time, data, *,
                 num_quant_bins=4):
        compartments = ("S", "I")  # R is implicit.
        duration = len(data)
        super().__init__(compartments, duration, population,
                         num_quant_bins=num_quant_bins)

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
        S_aux = (S0 - S2I.cumsum(-1)).clamp(min=0.5)
        # Account for the single initial infection.
        S2I[0] += 1
        # Assume infection lasts less than a month.
        recovery = torch.arange(30.).div(self.recovery_time).neg().exp()
        I_aux = convolve(S2I, recovery)[:len(self.data)].clamp(min=0.5)

        return {
            "R0": torch.tensor(2.0),
            "rho": torch.tensor(0.5),
            "auxiliary": torch.stack([S_aux, I_aux]),
        }

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Uniform(0, 1))

        # Convert interpretable parameters to distribution parameters.
        rate_s = -R0 / (tau * self.population)
        prob_i = 1 / tau

        return rate_s, prob_i, rho

    def initialize(self, params):
        # Start with a single infection.
        return {"S": self.population - 1, "I": 1}

    def transition_fwd(self, params, state, t):
        rate_s, prob_i, rho = params

        # Sample flows between compartments.
        prob_s = -(rate_s * state["I"]).expm1()
        S2I = pyro.sample("S2I_{}".format(t),
                          dist.Binomial(state["S"], prob_s))
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], prob_i))

        # Update compartements with flows.
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t] if t < self.duration else None)

    def transition_bwd(self, params, prev, curr, t):
        rate_s, prob_i, rho = params

        # Reverse the flow computation.
        S2I = prev["S"] - curr["S"]
        I2R = prev["I"] - curr["I"] + S2I

        # Condition on flows between compartments.
        prob_s = -(rate_s * prev["I"]).expm1()
        pyro.sample("S2I_{}".format(t),
                    dist.ExtendedBinomial(prev["S"], prob_s),
                    obs=S2I)
        pyro.sample("I2R_{}".format(t),
                    dist.ExtendedBinomial(prev["I"], prob_i),
                    obs=I2R)

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t])
