import torch

from tests.common import assert_equal
import pyro
from pyro.contrib.timeseries import IndependentMaternGP, LinearlyCoupledMaternGP
import pytest


@pytest.mark.parametrize('model,obs_dim', [('lcmgp', 3), ('imgp', 1), ('imgp', 3)])
@pytest.mark.parametrize('nu', [1.5, 2.5])
@pytest.mark.parametrize('T', [11, 37])
def test_independent_matern_gp(model, nu, obs_dim, T):
    torch.set_default_tensor_type('torch.DoubleTensor')
    dt = 0.1 + torch.rand(1).item()

    if model == 'lcmgp':
        num_gps = 2
        gp = LinearlyCoupledMaternGP(nu=nu, obs_dim=obs_dim, dt=dt, num_gps=num_gps,
                                     log_length_scale_init=torch.randn(num_gps),
                                     log_kernel_scale_init=torch.randn(num_gps),
                                     log_obs_noise_scale_init=torch.randn(obs_dim))
    elif model == 'imgp':
        gp = IndependentMaternGP(nu=nu, obs_dim=obs_dim, dt=dt,
                                 log_length_scale_init=torch.randn(obs_dim),
                                 log_kernel_scale_init=torch.randn(obs_dim),
                                 log_obs_noise_scale_init=torch.randn(obs_dim))

    targets = torch.randn(T, obs_dim)
    gp_log_prob = gp.log_prob(targets)
    if model == 'imgp':
        assert gp_log_prob.shape == (obs_dim,)
    else:
        assert gp_log_prob.dim() == 0

    if model == 'imgp':
        times = dt * torch.arange(T).double()
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

    for S in [1, 5]:
        dts = torch.rand(S).cumsum(dim=-1)
        predictive = gp.forecast(targets, dts)
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

    # the distant future
    dts = torch.tensor([500.0])
    predictive = gp.forecast(targets, dts)
    # assert mean reverting
    assert_equal(predictive.loc, torch.zeros(1, obs_dim))
