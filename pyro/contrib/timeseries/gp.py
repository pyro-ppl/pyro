import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import math

import pyro
import pyro.distributions as dist

from pyro.ops.ssm_gp import MaternKernel
from pyro.ops.tensor_utils import block_diag


class TimeSeriesModel(nn.Module):
    """
    Base class for univariate and multivariate time series models.
    """
    def log_prob(self, targets):
        """
        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape `(T, obs_dim)`, where `T` is the length of the time series and `obs_dim`
            is the dimension of the real-valued `targets` at each time step
        :returns torch.Tensor: A 0-dimensional log probability for the case of properly
            multivariate time series models in which the output dimensions are correlated;
            otherwise returns a 1-dimensional tensor of log probabilities for batched
            univariate time series models.
        """
        raise NotImplementedError

    def predict(self, targets, dts):
        """
        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape `(T, obs_dim)`, where `T` is the length of the time series and `obs_dim`
            is the dimension of the real-valued targets at each time step. These
            represent the training data that are conditioned on for the purpose of making
            forecasts.
        :param torch.Tensor dts: A 1-dimensional tensor of times to forecast into the future,
            with zero corresponding to the time of the final target `targets[-1]`.
        :returns tuple: Returns a predictive distribution with batch shape `(S,)` and
            event shape `(obs_dim,)`, where `S` is the size of `dts`. That is, the resulting
            predictive distributions do not encode correlations between distinct times in `dts`.
        """
        raise NotImplementedError



class IndependentMaternGP(TimeSeriesModel):
    """
    A time series model in which each output dimension is modeled independently
    with a univariate Gaussian Process with a Matern kernel. The targets are assumed
    to be evenly spaced in time. Training and inference are logarithmic in the length
    of the time series T.
    """
    def __init__(self, nu=1.5, dt=1.0, obs_dim=1,
                 log_length_scale_init=None, log_kernel_scale_init=None,
                 log_obs_noise_scale_init=None):
        """
        :param float nu: The order of the Matern kernel; either 1.5 or 2.5.
        :param float dt: The time spacing between neighboring observations of the time series.
        :param int obs_dim: The dimension of the targets at each time step.
        """
        self.nu = nu
        self.dt = dt
        self.obs_dim = obs_dim

        if log_obs_noise_scale_init is None:
            log_obs_noise_scale_init = -2.0 * torch.ones(obs_dim)
        assert log_obs_noise_scale_init.shape == (obs_dim,)

        super(IndependentMaternGP, self).__init__()

        self.kernel = MaternKernel(nu=nu, num_gps=obs_dim,
                                   log_length_scale_init=log_length_scale_init,
                                   log_kernel_scale_init=log_kernel_scale_init)

        self.log_obs_noise_scale = nn.Parameter(log_obs_noise_scale_init)

        obs_matrix = [1.0] + [0.0] * (self.kernel.state_dim - 1)
        self.register_buffer("obs_matrix", torch.tensor(obs_matrix).unsqueeze(-1))

    def _get_obs_noise_scale(self):
        return self.log_obs_noise_scale.exp()

    def _get_init_dist(self):
        return torch.distributions.MultivariateNormal(torch.zeros(self.obs_dim, self.kernel.state_dim),
                                                      self.kernel.stationary_covariance().squeeze(-3))

    def _get_obs_dist(self):
        return dist.Normal(torch.zeros(self.obs_dim, 1, 1),
                           self._get_obs_noise_scale().unsqueeze(-1).unsqueeze(-1)).to_event(1)

    def _get_dist(self):
        """
        Get the `GaussianHMM` distribution that corresponds to `obs_dim`-many independent Matern GPs.
        """
        trans_matrix, process_covar = self.kernel.transition_matrix_and_covariance(dt=self.dt)
        trans_dist = MultivariateNormal(torch.zeros(self.obs_dim, 1, self.kernel.state_dim),
                                        process_covar.unsqueeze(-3))
        trans_matrix = trans_matrix.unsqueeze(-3)
        return dist.GaussianHMM(self._get_init_dist(), trans_matrix, trans_dist,
                                self.obs_matrix, self._get_obs_dist())

    def log_prob(self, targets):
        """
        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape `(T, obs_dim)`, where `T` is the length of the time series and `obs_dim`
            is the dimension of the real-valued `targets` at each time step
        :returns torch.Tensor: A 1-dimensional tensor of log probabilities of shape `(obs_dim,)`
        """
        assert targets.dim() == 2 and targets.size(-1) == self.obs_dim
        return self._get_dist().log_prob(targets.t().unsqueeze(-1))

    def _filter(self, targets):
        """
        Return the filtering state for the associated state space model.
        """
        assert targets.dim() == 2 and targets.size(-1) == self.obs_dim
        return self._get_dist().filter(value.t().unsqueeze(-1))

    def _predict(self, dt, filtering_state, include_observation_noise=True):
        """
        Internal helper for prediction.
        """
        trans_mat = self.transition_matrix(dt).squeeze(-3)
        predicted_mean = torch.matmul(filtering_state.loc.unsqueeze(-2), trans_mat).squeeze(-2)[..., 0]
        predicted_function_covar = torch.matmul(trans_mat.transpose(-1, -2),
                                                torch.matmul(filtering_state.covariance_matrix,
                                                trans_mat))[..., 0, 0]

        if include_observation_noise:
            predicted_function_covar = predicted_function_covar + self.get_sigma_obs() ** 2
        return predicted_mean, predicted_function_covar
