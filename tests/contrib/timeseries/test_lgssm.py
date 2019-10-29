import torch

from tests.common import assert_equal
from pyro.contrib.timeseries import GenericLGSSM
import pytest


@pytest.mark.parametrize('state_dim', [2, 3])
@pytest.mark.parametrize('obs_dim', [2, 4])
@pytest.mark.parametrize('T', [11, 17])
def test_generic_lgssm_forecast(state_dim, obs_dim, T):
    torch.set_default_tensor_type('torch.DoubleTensor')

    model = GenericLGSSM(state_dim=state_dim, obs_dim=obs_dim,
                         log_obs_noise_scale_init=torch.randn(obs_dim))

    targets = torch.randn(T, obs_dim)
    filtering_state = model._filter(targets)

    actual_loc, actual_cov = model._forecast(3, filtering_state, include_observation_noise=False)

    obs_matrix = model.obs_matrix
    trans_matrix = model.trans_matrix
    trans_matrix_sq = torch.mm(trans_matrix, trans_matrix)
    trans_matrix_cubed = torch.mm(trans_matrix_sq, trans_matrix)

    trans_obs = torch.mm(trans_matrix, obs_matrix)
    trans_trans_obs = torch.mm(trans_matrix_sq, obs_matrix)
    trans_trans_trans_obs = torch.mm(trans_matrix_cubed, obs_matrix)

    predicted_mean1 = torch.mm(filtering_state.loc.unsqueeze(-2), trans_obs).squeeze(-2)
    predicted_mean2 = torch.mm(filtering_state.loc.unsqueeze(-2), trans_trans_obs).squeeze(-2)
    predicted_mean3 = torch.mm(filtering_state.loc.unsqueeze(-2), trans_trans_trans_obs).squeeze(-2)

    # check predicted means for 3 timesteps
    assert_equal(actual_loc[0], predicted_mean1)
    assert_equal(actual_loc[1], predicted_mean2)
    assert_equal(actual_loc[2], predicted_mean3)

    # check predicted covariances for 3 timesteps
    process_covar = model._get_trans_dist().covariance_matrix
    fs_covar = filtering_state.covariance_matrix

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
