# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pyro
import pyro.distributions as dist

from .compartmental import CompartmentalModel


class SIRModel(CompartmentalModel):
    def __init__(self, population, recovery_time, data):
        compartments = ("S", "I")  # R is implicit.
        duration = len(data)
        super().__init__(self, compartments, duration, population)

        assert isinstance(recovery_time, float)
        assert recovery_time > 0
        self.recovery_time = recovery_time

        self.data = data

    def global_model(self):
        tau = self.recovery_time
        R0 = pyro.sample("R0", dist.LogNormal(0., 1.))
        rho = pyro.sample("rho", dist.Uniform(0, 1))

        # Convert interpretable parameters to distribution parameters.
        rate_s = -R0 / (tau * self.population)
        prob_i = 1 / (1 + tau)

        return rate_s, prob_i, rho

    def initialize(self, params):
        return {"S": 1. - self.population, "I": 1.}

    def transition_fwd(self, params, state, t):
        rate_s, prob_i, rho = params

        # Compute state update.
        prob_s = -(rate_s * state["I"]).expm1()
        S2I = pyro.sample("S2I_{}".format(t),
                          dist.Binomial(state["S"], prob_s))
        I2R = pyro.sample("I2R_{}".format(t),
                          dist.Binomial(state["I"], prob_i))
        state["S"] = state["S"] - S2I
        state["I"] = state["I"] + S2I - I2R

        # Condition on observations.
        pyro.sample("obs_{}".format(t),
                    dist.ExtendedBinomial(S2I, rho),
                    obs=self.data[t] if t < self.duration else None)

    def transition_bwd(self, params, prev, curr):
        rate_s, prob_i, rho = params

        # Reverse the S2I,I2R computation.
        S2I = prev["S"] - curr["S"]
        I2R = prev["I"] - curr["I"] + S2I

        # Compute probability factors.
        prob_s = -(rate_s * prev["I"]).expm1()
        S2I_logp = dist.ExtendedBinomial(prev["S"], prob_s).log_prob(S2I)
        I2R_logp = dist.ExtendedBinomial(prev["I"], prob_i).log_prob(I2R)
        # FIXME the following line needs to .unsqueeze() data for enumeration.
        obs_logp = dist.ExtendedBinomial(S2I.clamp(min=0), rho).log_prob(self.data)
        return obs_logp + S2I_logp + I2R_logp
