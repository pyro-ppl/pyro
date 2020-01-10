# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import config_enumerate
from pyro.ops.indexing import Vindex


def guide_generic(config):
    """generic mean-field guide for continuous random effects"""
    N_state = config["sizes"]["state"]

    if config["group"]["random"] == "continuous":
        loc_g = pyro.param("loc_group", lambda: torch.zeros((N_state ** 2,)))
        scale_g = pyro.param("scale_group", lambda: torch.ones((N_state ** 2,)),
                             constraint=constraints.positive)

    # initialize individual-level random effect parameters
    N_c = config["sizes"]["group"]
    if config["individual"]["random"] == "continuous":
        loc_i = pyro.param("loc_individual", lambda: torch.zeros((N_c, N_state ** 2,)))
        scale_i = pyro.param("scale_individual", lambda: torch.ones((N_c, N_state ** 2,)),
                             constraint=constraints.positive)

    N_c = config["sizes"]["group"]
    with pyro.plate("group", N_c, dim=-1):

        if config["group"]["random"] == "continuous":
            pyro.sample("eps_g", dist.Normal(loc_g, scale_g).to_event(1),
                        )  # infer={"num_samples": 10})

        N_s = config["sizes"]["individual"]
        with pyro.plate("individual", N_s, dim=-2), poutine.mask(mask=config["individual"]["mask"]):

            # individual-level random effects
            if config["individual"]["random"] == "continuous":
                pyro.sample("eps_i", dist.Normal(loc_i, scale_i).to_event(1),
                            )  # infer={"num_samples": 10})


@config_enumerate
def model_generic(config):
    """Hierarchical mixed-effects hidden markov model"""

    MISSING = config["MISSING"]
    N_v = config["sizes"]["random"]
    N_state = config["sizes"]["state"]

    # initialize group-level random effect parameterss
    if config["group"]["random"] == "discrete":
        probs_e_g = pyro.param("probs_e_group", lambda: torch.randn((N_v,)).abs(), constraint=constraints.simplex)
        theta_g = pyro.param("theta_group", lambda: torch.randn((N_v, N_state ** 2)))
    elif config["group"]["random"] == "continuous":
        loc_g = torch.zeros((N_state ** 2,))
        scale_g = torch.ones((N_state ** 2,))

    # initialize individual-level random effect parameters
    N_c = config["sizes"]["group"]
    if config["individual"]["random"] == "discrete":
        probs_e_i = pyro.param("probs_e_individual",
                               lambda: torch.randn((N_c, N_v,)).abs(),
                               constraint=constraints.simplex)
        theta_i = pyro.param("theta_individual",
                             lambda: torch.randn((N_c, N_v, N_state ** 2)))
    elif config["individual"]["random"] == "continuous":
        loc_i = torch.zeros((N_c, N_state ** 2,))
        scale_i = torch.ones((N_c, N_state ** 2,))

    # initialize likelihood parameters
    # observation 1: step size (step ~ Gamma)
    step_zi_param = pyro.param("step_zi_param", lambda: torch.ones((N_state, 2)))
    step_concentration = pyro.param("step_param_concentration",
                                    lambda: torch.randn((N_state,)).abs(),
                                    constraint=constraints.positive)
    step_rate = pyro.param("step_param_rate",
                           lambda: torch.randn((N_state,)).abs(),
                           constraint=constraints.positive)

    # observation 2: step angle (angle ~ VonMises)
    angle_concentration = pyro.param("angle_param_concentration",
                                     lambda: torch.randn((N_state,)).abs(),
                                     constraint=constraints.positive)
    angle_loc = pyro.param("angle_param_loc", lambda: torch.randn((N_state,)).abs())

    # observation 3: dive activity (omega ~ Beta)
    omega_zi_param = pyro.param("omega_zi_param", lambda: torch.ones((N_state, 2)))
    omega_concentration0 = pyro.param("omega_param_concentration0",
                                      lambda: torch.randn((N_state,)).abs(),
                                      constraint=constraints.positive)
    omega_concentration1 = pyro.param("omega_param_concentration1",
                                      lambda: torch.randn((N_state,)).abs(),
                                      constraint=constraints.positive)

    # initialize gamma to uniform
    gamma = torch.zeros((N_state ** 2,))

    N_c = config["sizes"]["group"]
    with pyro.plate("group", N_c, dim=-1):

        # group-level random effects
        if config["group"]["random"] == "discrete":
            # group-level discrete effect
            e_g = pyro.sample("e_g", dist.Categorical(probs_e_g))
            eps_g = Vindex(theta_g)[..., e_g, :]
        elif config["group"]["random"] == "continuous":
            eps_g = pyro.sample("eps_g", dist.Normal(loc_g, scale_g).to_event(1),
                                )  # infer={"num_samples": 10})
        else:
            eps_g = 0.

        # add group-level random effect to gamma
        gamma = gamma + eps_g

        N_s = config["sizes"]["individual"]
        with pyro.plate("individual", N_s, dim=-2), poutine.mask(mask=config["individual"]["mask"]):

            # individual-level random effects
            if config["individual"]["random"] == "discrete":
                # individual-level discrete effect
                e_i = pyro.sample("e_i", dist.Categorical(probs_e_i))
                eps_i = Vindex(theta_i)[..., e_i, :]
                # assert eps_i.shape[-3:] == (1, N_c, N_state ** 2) and eps_i.shape[0] == N_v
            elif config["individual"]["random"] == "continuous":
                eps_i = pyro.sample("eps_i", dist.Normal(loc_i, scale_i).to_event(1),
                                    )  # infer={"num_samples": 10})
            else:
                eps_i = 0.

            # add individual-level random effect to gamma
            gamma = gamma + eps_i

            y = torch.tensor(0).long()

            N_t = config["sizes"]["timesteps"]
            for t in pyro.markov(range(N_t)):
                with poutine.mask(mask=config["timestep"]["mask"][..., t]):
                    gamma_t = gamma  # per-timestep variable

                    # finally, reshape gamma as batch of transition matrices
                    gamma_t = gamma_t.reshape(tuple(gamma_t.shape[:-1]) + (N_state, N_state))

                    # we've accounted for all effects, now actually compute gamma_y
                    gamma_y = Vindex(gamma_t)[..., y, :]
                    y = pyro.sample("y_{}".format(t), dist.Categorical(logits=gamma_y))

                    # observation 1: step size
                    step_dist = dist.Gamma(
                        concentration=Vindex(step_concentration)[..., y],
                        rate=Vindex(step_rate)[..., y]
                    )

                    # zero-inflation with MaskedMixture
                    step_zi = Vindex(step_zi_param)[..., y, :]
                    step_zi_mask = pyro.sample("step_zi_{}".format(t),
                                               dist.Categorical(logits=step_zi),
                                               obs=(config["observations"]["step"][..., t] == MISSING))
                    step_zi_zero_dist = dist.Delta(v=torch.tensor(MISSING))
                    step_zi_dist = dist.MaskedMixture(step_zi_mask, step_dist, step_zi_zero_dist)

                    pyro.sample("step_{}".format(t),
                                step_zi_dist,
                                obs=config["observations"]["step"][..., t])

                    # observation 2: step angle
                    angle_dist = dist.VonMises(
                        concentration=Vindex(angle_concentration)[..., y],
                        loc=Vindex(angle_loc)[..., y]
                    )
                    pyro.sample("angle_{}".format(t),
                                angle_dist,
                                obs=config["observations"]["angle"][..., t])

                    # observation 3: dive activity
                    omega_dist = dist.Beta(
                        concentration0=Vindex(omega_concentration0)[..., y],
                        concentration1=Vindex(omega_concentration1)[..., y]
                    )

                    # zero-inflation with MaskedMixture
                    omega_zi = Vindex(omega_zi_param)[..., y, :]
                    omega_zi_mask = pyro.sample(
                        "omega_zi_{}".format(t),
                        dist.Categorical(logits=omega_zi),
                        obs=(config["observations"]["omega"][..., t] == MISSING))

                    omega_zi_zero_dist = dist.Delta(v=torch.tensor(MISSING))
                    omega_zi_dist = dist.MaskedMixture(omega_zi_mask, omega_dist, omega_zi_zero_dist)

                    pyro.sample("omega_{}".format(t),
                                omega_zi_dist,
                                obs=config["observations"]["omega"][..., t])
