# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import torch

from tests.common import assert_equal
import pyro
from pyro.contrib.timeseries import (IndependentMaternGP, LinearlyCoupledMaternGP, GenericLGSSM,
                                     GenericLGSSMWithGPNoiseModel, DependentMaternGP)
from pyro.ops.tensor_utils import block_diag_embed
import pytest


@pytest.mark.parametrize('model,obs_dim,nu_statedim', [('ssmgp', 3, 1.5), ('ssmgp', 2, 2.5),
                                                       ('lcmgp', 3, 1.5), ('lcmgp', 2, 2.5),
                                                       ('imgp', 1, 0.5), ('imgp', 2, 0.5),
                                                       ('imgp', 1, 1.5), ('imgp', 3, 1.5),
                                                       ('imgp', 1, 2.5), ('imgp', 3, 2.5),
                                                       ('dmgp', 1, 1.5), ('dmgp', 2, 1.5),
                                                       ('dmgp', 3, 1.5),
                                                       ('glgssm', 1, 3), ('glgssm', 3, 1)])
@pytest.mark.parametrize('T', [11, 37])
def test_timeseries_models(model, nu_statedim, obs_dim, T):
    torch.set_default_tensor_type('torch.DoubleTensor')
    dt = 0.1 + torch.rand(1).item()

    if model == 'lcmgp':
        num_gps = 2
        gp = LinearlyCoupledMaternGP(nu=nu_statedim, obs_dim=obs_dim, dt=dt, num_gps=num_gps,
                                     length_scale_init=0.5 + torch.rand(num_gps),
                                     kernel_scale_init=0.5 + torch.rand(num_gps),
                                     obs_noise_scale_init=0.5 + torch.rand(obs_dim))
    elif model == 'imgp':
        gp = IndependentMaternGP(nu=nu_statedim, obs_dim=obs_dim, dt=dt,
                                 length_scale_init=0.5 + torch.rand(obs_dim),
                                 kernel_scale_init=0.5 + torch.rand(obs_dim),
                                 obs_noise_scale_init=0.5 + torch.rand(obs_dim))
    elif model == 'glgssm':
        gp = GenericLGSSM(state_dim=nu_statedim, obs_dim=obs_dim,
                          obs_noise_scale_init=0.5 + torch.rand(obs_dim))
    elif model == 'ssmgp':
        state_dim = {0.5: 4, 1.5: 3, 2.5: 2}[nu_statedim]
        gp = GenericLGSSMWithGPNoiseModel(nu=nu_statedim, state_dim=state_dim, obs_dim=obs_dim,
                                          obs_noise_scale_init=0.5 + torch.rand(obs_dim))
    elif model == 'dmgp':
        linearly_coupled = bool(torch.rand(1).item() > 0.5)
        gp = DependentMaternGP(nu=nu_statedim, obs_dim=obs_dim, dt=dt, linearly_coupled=linearly_coupled,
                               length_scale_init=0.5 + torch.rand(obs_dim))

    targets = torch.randn(T, obs_dim)
    gp_log_prob = gp.log_prob(targets)
    if model == 'imgp':
        assert gp_log_prob.shape == (obs_dim,)
    else:
        assert gp_log_prob.dim() == 0

    # compare matern log probs to vanilla GP result via multivariate normal
    if model == 'imgp':
        times = dt * torch.arange(T).double()
        for dim in range(obs_dim):
            lengthscale = gp.kernel.length_scale[dim]
            variance = gp.kernel.kernel_scale.pow(2)[dim]
            obs_noise = gp.obs_noise_scale.pow(2)[dim]

            kernel = {0.5: pyro.contrib.gp.kernels.Exponential,
                      1.5: pyro.contrib.gp.kernels.Matern32,
                      2.5: pyro.contrib.gp.kernels.Matern52}[nu_statedim]
            kernel = kernel(input_dim=1, lengthscale=lengthscale, variance=variance)
            # XXX kernel(times) loads old parameters from param store
            kernel = kernel.forward(times) + obs_noise * torch.eye(T)

            mvn = torch.distributions.MultivariateNormal(torch.zeros(T), kernel)
            mvn_log_prob = mvn.log_prob(targets[:, dim])
            assert_equal(mvn_log_prob, gp_log_prob[dim], prec=1e-4)

    for S in [1, 5]:
        if model in ['imgp', 'lcmgp', 'dmgp', 'lcdgp']:
            dts = torch.rand(S).cumsum(dim=-1)
            predictive = gp.forecast(targets, dts)
        else:
            predictive = gp.forecast(targets, S)
        assert predictive.loc.shape == (S, obs_dim)
        if model == 'imgp':
            assert predictive.scale.shape == (S, obs_dim)
            # assert monotonic increase of predictive noise
            if S > 1:
                delta = predictive.scale[1:S, :] - predictive.scale[0:S-1, :]
                assert (delta > 0.0).sum() == (S - 1) * obs_dim
        else:
            assert predictive.covariance_matrix.shape == (S, obs_dim, obs_dim)
            # assert monotonic increase of predictive noise
            if S > 1:
                dets = predictive.covariance_matrix.det()
                delta = dets[1:S] - dets[0:S-1]
                assert (delta > 0.0).sum() == (S - 1)

    if model in ['imgp', 'lcmgp', 'dmgp', 'lcdgp']:
        # the distant future
        dts = torch.tensor([500.0])
        predictive = gp.forecast(targets, dts)
        # assert mean reverting behavior for GP models
        assert_equal(predictive.loc, torch.zeros(1, obs_dim))


@pytest.mark.parametrize('obs_dim', [1, 3])
def test_dependent_matern_gp(obs_dim):
    dt = 0.5 + torch.rand(1).item()
    gp = DependentMaternGP(nu=1.5, obs_dim=obs_dim, dt=dt,
                           length_scale_init=0.5 + torch.rand(obs_dim))

    # make sure stationary covariance matrix satisfies the relevant
    # matrix riccati equation
    lengthscale = gp.kernel.length_scale.unsqueeze(-1).unsqueeze(-1)
    F = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    mask1 = torch.tensor([[0.0, 0.0], [-3.0, 0.0]])
    mask2 = torch.tensor([[0.0, 0.0], [0.0, -math.sqrt(12.0)]])
    F = block_diag_embed(F + mask1 / lengthscale.pow(2.0) + mask2 / lengthscale)

    stat_cov = gp._stationary_covariance()
    wiener_cov = gp._get_wiener_cov()
    wiener_cov *= torch.tensor([[0.0, 0.0], [0.0, 1.0]]).repeat(obs_dim, obs_dim)

    expected_zero = torch.matmul(F, stat_cov) + torch.matmul(stat_cov, F.transpose(-1, -2)) + wiener_cov
    assert_equal(expected_zero, torch.zeros(gp.full_state_dim, gp.full_state_dim))
