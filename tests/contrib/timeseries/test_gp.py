import torch

from tests.common import assert_equal
import pyro
from pyro.contrib.timeseries import IndependentMaternGP, CoupledMaternGP
import pytest


@pytest.mark.parametrize('model', ['cmgp', 'imgp'])
@pytest.mark.parametrize('nu', [1.5, 2.5])
@pytest.mark.parametrize('obs_dim', [1, 3])
@pytest.mark.parametrize('T', [11, 37])
def test_independent_matern_gp(model, nu, obs_dim, T):
    torch.set_default_tensor_type('torch.DoubleTensor')
    dt = 0.1 + torch.rand(1).item()

    if model=='cmgp':
        if obs_dim==1:
            return
        num_gps = 2
        gp = CoupledMaternGP(nu=nu, obs_dim=obs_dim, dt=dt, num_gps=num_gps,
                             log_length_scale_init=torch.randn(num_gps),
                             log_kernel_scale_init=torch.randn(num_gps),
                             log_obs_noise_scale_init=torch.randn(obs_dim))
    elif model=='imgp':
        gp = IndependentMaternGP(nu=nu, obs_dim=obs_dim, dt=dt,
                                 log_length_scale_init=torch.randn(obs_dim),
                                 log_kernel_scale_init=torch.randn(obs_dim),
                                 log_obs_noise_scale_init=torch.randn(obs_dim))

    targets = torch.randn(T, obs_dim)
    gp_log_prob = gp.log_prob(targets)
    if model=='imgp':
        assert gp_log_prob.shape == (obs_dim,)
    else:
        assert gp_log_prob.dim() == 0

    times = dt * torch.arange(T).double()

    if model=='imgp':
        for dim in range(obs_dim):
            lengthscale = gp.kernel.log_length_scale.exp()[dim]
            variance = (2.0 * gp.kernel.log_kernel_scale).exp()[dim]
            obs_noise = (2.0 * gp.log_obs_noise_scale).exp()[dim]

            kernel = pyro.contrib.gp.kernels.Matern32 if nu == 1.5 else pyro.contrib.gp.kernels.Matern52
            kernel = kernel(input_dim=1, lengthscale=lengthscale, variance=variance)
            kernel = kernel(times) + obs_noise * torch.eye(T)

            mvn = torch.distributions.MultivariateNormal(torch.zeros(T), kernel)
            mvn_log_prob = mvn.log_prob(targets[:, dim])
            assert_equal(mvn_log_prob, gp_log_prob[dim])

    for S in [1, 3]:
        dts = torch.rand(S)
        predictive = gp.forecast(targets, dts)
        assert predictive.loc.shape == (S, obs_dim)
        if model=='imgp':
            assert predictive.scale.shape == (S, obs_dim)
        else:
            assert predictive.covariance_matrix.shape == (S, obs_dim, obs_dim)

    # the distant future
    dts = torch.tensor([500.0])
    predictive = gp.forecast(targets, dts)
    # assert mean reverting
    assert_equal(predictive.loc, torch.zeros(1, obs_dim))
    # assert large time covariance
    if model=='imgp':
        expected_scale = gp._get_obs_noise_scale().unsqueeze(0)
        assert_equal(predictive.scale, expected_scale)
    else:
        expected_cov = torch.eye(obs_dim) * gp._get_obs_noise_scale().pow(2.0)
        expected_cov = expected_cov.unsqueeze(0)
        assert_equal(predictive.covariance_matrix, expected_cov)
