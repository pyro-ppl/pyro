from __future__ import absolute_import, division, print_function

import logging

import numpy as np

import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import config_enumerate


def _index_param(param, ind, dim=-2):
    """helper for advanced indexing black magic"""
    # assume: dim < 0
    # assume: param.shape[dim:] == event_shape
    # assume: index.shape == batch_shape
    # assume: param.shape == batch_shape + event_shape
    # goal: slice into an event_dim with index
    # step 1: unsqueeze event dims in index
    for d in range(len(param.shape[dim:])):
        ind = ind.unsqueeze(-1)
    # step 2: generate dummy indices for all other dimensions of param
    inds = [None] * len(param.shape)
    for d, sd in enumerate(reversed(param.shape)):
        if dim == -d-1:
            inds[-d-1] = ind
        else:
            inds[-d-1] = torch.arange(sd).reshape((sd,) + (1,) * d)
    # step 3: use the index and dummy indices to select
    res = param[tuple(inds)]
    # XXX is this necessary?
    # step 4: squeeze out the empty event_dim
    return res.squeeze(dim)


def guide_generic(config):
    """generic mean-field guide for continuous random effects"""
    N_v = config["sizes"]["random"]
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
    with pyro.plate("group", N_c) as c:

        if config["group"]["random"] == "continuous":
            eps_g = pyro.sample("eps_g", dist.Normal(loc_g, scale_g).to_event(1),
                                )  # infer={"num_samples": 10})

        N_s = config["sizes"]["individual"]
        with pyro.plate("individual", N_s) as s, poutine.mask(mask=config["individual"]["mask"]):

            # individual-level random effects
            if config["individual"]["random"] == "continuous":
                eps_i = pyro.sample("eps_i", dist.Normal(loc_i, scale_i).to_event(1),
                                    )  # infer={"num_samples": 10})


@config_enumerate
def model_generic(config):
    """generic hierarchical mixed-effects hidden markov model"""

    N_v = config["sizes"]["random"]
    N_state = config["sizes"]["state"]

    # initialize fixed effect parameters - all the same size
    if config["group"]["fixed"] is not None:
        N_fg = config["group"]["fixed"].shape[-1]
        beta_g = pyro.param("beta_group", lambda: torch.ones((N_fg, N_state ** 2)))

    if config["individual"]["fixed"] is not None:
        N_fi = config["individual"]["fixed"].shape[-1]
        beta_i = pyro.param("beta_individual", lambda: torch.ones((N_fi, N_state ** 2)))

    if config["timestep"]["fixed"] is not None:
        N_ft = config["timestep"]["fixed"].shape[-1]
        beta_t = pyro.param("beta_timestep", lambda: torch.ones((N_ft, N_state ** 2)))

    # initialize group-level random effect parameterss
    if config["group"]["random"] == "discrete":
        probs_e_g = pyro.param("probs_e_group", lambda: torch.randn((N_v,)).abs(), constraint=constraints.simplex)
        theta_g = pyro.param("theta_group", lambda: torch.randn((N_v, N_state ** 2)))
    elif config["group"]["random"] == "continuous":
        loc_g = torch.zeros((N_state ** 2,))
        scale_g = torch.ones((N_state ** 2,))
    else:  # none
        pass

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
    else:  # none
        pass

    # initialize likelihood parameters
    for coord, coord_config in config["observations"].items():
        if coord_config["zi"]:
            pyro.param("{}_zi_param".format(coord), lambda: torch.ones((N_state,2)))
        for arg_name, arg_constraint in coord_config["dist"].arg_constraints.items():
            pyro.param("{}_param_{}".format(coord, arg_name),
                       lambda: torch.randn((N_state,)).abs(),
                       constraint=arg_constraint)

    # initialize gamma to uniform
    gamma = torch.zeros((N_state ** 2,))

    N_c = config["sizes"]["group"]
    with pyro.plate("group", N_c, dim=-1) as c:

        # group-level random effects
        if config["group"]["random"] == "discrete":
            # group-level discrete effect
            e_g = pyro.sample("e_g", dist.Categorical(probs_e_g))
            eps_g = _index_param(theta_g, e_g, dim=-2)
        elif config["group"]["random"] == "continuous":
            eps_g = pyro.sample("eps_g", dist.Normal(loc_g, scale_g).to_event(1),
                                )  # infer={"num_samples": 10})
        else:
            eps_g = 0.

        # add group-level random effect to gamma
        gamma = gamma + eps_g

        # group-level fixed effects
        if config["group"]["fixed"] is not None:
            covariates_g = config["individual"]["fixed"]
            beta_g = pyro.param("beta_group")
            fixed_g = torch.einsum("...f,fs->...s",
                                   [covariates_g, beta_g])
            gamma = gamma + fixed_g

        N_s = config["sizes"]["individual"]
        with pyro.plate("individual", N_s, dim=-2) as s, poutine.mask(mask=config["individual"]["mask"]):

            # individual-level random effects
            if config["individual"]["random"] == "discrete":
                # individual-level discrete effect
                e_i = pyro.sample("e_i", dist.Categorical(probs_e_i))
                eps_i = _index_param(theta_i, e_i, dim=-2)
                assert eps_i.shape[-3:] == (1, N_c, N_state ** 2) and eps_i.shape[0] == N_v
            elif config["individual"]["random"] == "continuous":
                eps_i = pyro.sample("eps_i", dist.Normal(loc_i, scale_i).to_event(1),
                                    )  # infer={"num_samples": 10})
            else:
                eps_i = 0.

            # add individual-level random effect to gamma
            gamma = gamma + eps_i
            assert gamma.shape == (eps_g + eps_i).shape

            # individual-level fixed effects
            if config["individual"]["fixed"] is not None:
                covariates_i = config["individual"]["fixed"]
                beta_i = pyro.param("beta_individual")
                fixed_i =  torch.einsum("...f,fs->...s",
                                        [covariates_i, beta_i])
                gamma = gamma + fixed_i

            # TODO initialize y from stationary distribution?
            y = torch.tensor(0).long()

            N_t = config["sizes"]["timesteps"]
            for t in pyro.markov(range(N_t)):
                with poutine.mask(mask=config["timestep"]["mask"][..., t]):
                    # per-timestep fixed effects
                    gamma_t = gamma  # per-timestep variable
                    if config["timestep"]["fixed"] is not None:
                        covariates_t = config["timestep"]["fixed"][..., t, :]
                        beta_t = pyro.param("beta_timestep")
                        fixed_t = torch.einsum("...f,fs->...s",
                                               [covariates_t, beta_t])
                        gamma_t = gamma_t + fixed_t

                    # finally, reshape gamma as batch of transition matrices
                    gamma_t = gamma_t.reshape(tuple(gamma_t.shape[:-1]) + (N_state, N_state))

                    # we've accounted for all effects, now actually compute gamma_y
                    # gamma_y = _index_gamma(gamma_t, y, t)
                    gamma_y = _index_param(gamma_t, y, dim=-2)
                    y = pyro.sample("y_{}".format(t), dist.Categorical(logits=gamma_y))

                    # multivariate observations with different distributions
                    for coord, coord_config in config["observations"].items():
                        coord_params = [
                            _index_param(pyro.param("{}_param_{}".format(coord, arg_name)), y, dim=-1)
                            for arg_name in coord_config["dist"].arg_constraints.keys()
                        ]
                        coord_dist = coord_config["dist"](*coord_params)

                        if not coord_config["zi"]:
                            pyro.sample("{}_{}".format(coord, t), 
                                        coord_dist,
                                        obs=coord_config["values"][..., t])
                        elif coord_config["zi"]:
                            # zero-inflation with MaskedMixture
                            coord_zi = _index_param(pyro.param("{}_zi_param".format(coord)), y, dim=-2)
                            # coord_zi_mask = coord_config["values"][..., t] == 1e-4
                            # coord_zi_scale = dist.Categorical(logits=coord_zi).log_prob(coord_zi_mask).exp()
                            coord_zi_mask = pyro.sample("{}_zi_{}".format(coord, t),
                                                        dist.Categorical(logits=coord_zi), 
                                                        obs=(coord_config["values"][..., t] == 1e-4))
                            coord_zi_zero_dist = dist.Delta(v=torch.tensor(1e-4))
                            coord_zi_dist = dist.MaskedMixture(coord_zi_mask, coord_dist, coord_zi_zero_dist)

                            # do a bit of gross nan error checking...
                            # if t > 5 and t < 10:
                            #     nan_check_mask = config["timestep"]["mask"][..., t] & config["individual"]["mask"]
                            #     assert not torch.isnan(coord_zi_dist.log_prob(coord_config["values"][..., t]).sum(dim=0).squeeze()[nan_check_mask]).any(), \
                            #         "nan zi at {}_{}".format(coord, t)

                            #     assert not (coord_zi_dist.log_prob(coord_config["values"][..., t]).sum(dim=0).squeeze()[nan_check_mask] == 0.).all(), \
                            #         "zero zi at {}_{}".format(coord, t)

                            pyro.sample("{}_{}".format(coord, t), 
                                        coord_zi_dist,
                                        obs=coord_config["values"][..., t])
