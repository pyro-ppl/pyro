# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from tests.common import assert_equal
from pyro.contrib.timeseries import GenericLGSSM, GenericLGSSMWithGPNoiseModel
import pytest


@pytest.mark.parametrize('model_class', ['lgssm', 'lgssmgp'])
@pytest.mark.parametrize('state_dim', [2, 3])
@pytest.mark.parametrize('obs_dim', [2, 4])
@pytest.mark.parametrize('T', [11, 17])
def test_generic_lgssm_forecast(model_class, state_dim, obs_dim, T):
    torch.set_default_tensor_type('torch.DoubleTensor')

    if model_class == 'lgssm':
        model = GenericLGSSM(state_dim=state_dim, obs_dim=obs_dim,
                             obs_noise_scale_init=0.1 + torch.rand(obs_dim))
    elif model_class == 'lgssmgp':
        model = GenericLGSSMWithGPNoiseModel(state_dim=state_dim, obs_dim=obs_dim, nu=1.5,
                                             obs_noise_scale_init=0.1 + torch.rand(obs_dim))
        # with these hyperparameters we essentially turn off the GP contributions
        model.kernel.length_scale = 1.0e-6 * torch.ones(obs_dim)
        model.kernel.kernel_scale = 1.0e-6 * torch.ones(obs_dim)

    targets = torch.randn(T, obs_dim)
    filtering_state = model._filter(targets)

    actual_loc, actual_cov = model._forecast(3, filtering_state, include_observation_noise=False)

    obs_matrix = model.obs_matrix if model_class == 'lgssm' else model.z_obs_matrix
    trans_matrix = model.trans_matrix if model_class == 'lgssm' else model.z_trans_matrix
    trans_matrix_sq = torch.mm(trans_matrix, trans_matrix)
    trans_matrix_cubed = torch.mm(trans_matrix_sq, trans_matrix)

    trans_obs = torch.mm(trans_matrix, obs_matrix)
    trans_trans_obs = torch.mm(trans_matrix_sq, obs_matrix)
    trans_trans_trans_obs = torch.mm(trans_matrix_cubed, obs_matrix)

    # we only compute contributions for the state space portion for lgssmgp
    fs_loc = filtering_state.loc if model_class == 'lgssm' else filtering_state.loc[-state_dim:]

    predicted_mean1 = torch.mm(fs_loc.unsqueeze(-2), trans_obs).squeeze(-2)
    predicted_mean2 = torch.mm(fs_loc.unsqueeze(-2), trans_trans_obs).squeeze(-2)
    predicted_mean3 = torch.mm(fs_loc.unsqueeze(-2), trans_trans_trans_obs).squeeze(-2)

    # check predicted means for 3 timesteps
    assert_equal(actual_loc[0], predicted_mean1)
    assert_equal(actual_loc[1], predicted_mean2)
    assert_equal(actual_loc[2], predicted_mean3)

    # check predicted covariances for 3 timesteps
    fs_covar, process_covar = None, None
    if model_class == 'lgssm':
        process_covar = model._get_trans_dist().covariance_matrix
        fs_covar = filtering_state.covariance_matrix
    elif model_class == 'lgssmgp':
        # we only compute contributions for the state space portion
        process_covar = model.trans_noise_scale_sq.diag_embed()
        fs_covar = filtering_state.covariance_matrix[-state_dim:, -state_dim:]

    predicted_covar1 = torch.mm(trans_obs.t(), torch.mm(fs_covar, trans_obs)) + \
        torch.mm(obs_matrix.t(), torch.mm(process_covar, obs_matrix))

    predicted_covar2 = torch.mm(trans_trans_obs.t(), torch.mm(fs_covar, trans_trans_obs)) + \
        torch.mm(trans_obs.t(), torch.mm(process_covar, trans_obs)) + \
        torch.mm(obs_matrix.t(), torch.mm(process_covar, obs_matrix))

    predicted_covar3 = torch.mm(trans_trans_trans_obs.t(), torch.mm(fs_covar, trans_trans_trans_obs)) + \
        torch.mm(trans_trans_obs.t(), torch.mm(process_covar, trans_trans_obs)) + \
        torch.mm(trans_obs.t(), torch.mm(process_covar, trans_obs)) + \
        torch.mm(obs_matrix.t(), torch.mm(process_covar, obs_matrix))

    assert_equal(actual_cov[0], predicted_covar1)
    assert_equal(actual_cov[1], predicted_covar2)
    assert_equal(actual_cov[2], predicted_covar3)
