import torch

from tests.common import assert_equal
import pyro
from pyro.contrib.timeseries import IndependentMaternGP
import pytest


@pytest.mark.parametrize('nu', [1.5, 2.5])
@pytest.mark.parametrize('obs_dim', [1, 3])
@pytest.mark.parametrize('T', [11, 37])
def test_independent_matern_gp(nu, obs_dim, T):
    torch.set_default_tensor_type('torch.DoubleTensor')
    dt = 0.1 + torch.rand(1).item()

    gp = IndependentMaternGP(nu=nu, obs_dim=obs_dim, dt=dt,
                             log_length_scale_init=torch.randn(obs_dim),
                             log_kernel_scale_init=torch.randn(obs_dim),
                             log_obs_noise_scale_init=torch.randn(obs_dim))
    targets = torch.randn(T, obs_dim)
    gp_log_prob = gp.log_prob(targets)
    assert gp_log_prob.shape == (obs_dim,)

    times = dt * torch.arange(T).double()

    for dim in range(obs_dim):
        lengthscale = gp.kernel.log_length_scale.exp()[dim]
        variance = (2.0 * gp.kernel.log_kernel_scale).exp()[dim]
        obs_noise = (2.0 * gp.log_obs_noise_scale).exp()[dim]

        kernel = pyro.contrib.gp.kernels.Matern32 if nu==1.5 else pyro.contrib.gp.kernels.Matern52
        kernel = kernel(input_dim=1, lengthscale=lengthscale, variance=variance)
        kernel = kernel(times) + obs_noise * torch.eye(T)

        mvn = torch.distributions.MultivariateNormal(torch.zeros(T), kernel)
        mvn_log_prob = mvn.log_prob(targets[:, dim])
        assert_equal(mvn_log_prob, gp_log_prob[dim])
