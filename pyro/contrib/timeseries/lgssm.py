# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, constraints

import pyro.distributions as dist
from pyro.contrib.timeseries.base import TimeSeriesModel
from pyro.nn import PyroParam, pyro_method
from pyro.ops.tensor_utils import repeated_matmul


class GenericLGSSM(TimeSeriesModel):
    """
    A generic Linear Gaussian State Space Model parameterized with arbitrary time invariant
    transition and observation dynamics. The targets are (implicitly) assumed to be evenly
    spaced in time. Training and inference are logarithmic in the length of the time series T.

    :param int obs_dim: The dimension of the targets at each time step.
    :param int state_dim: The dimension of latent state at each time step.
    :param bool learnable_observation_loc: whether the mean of the observation model should be learned or not;
        defaults to False.
    """
    def __init__(self, obs_dim=1, state_dim=2, obs_noise_scale_init=None,
                 learnable_observation_loc=False):
        self.obs_dim = obs_dim
        self.state_dim = state_dim

        if obs_noise_scale_init is None:
            obs_noise_scale_init = 0.2 * torch.ones(obs_dim)
        assert obs_noise_scale_init.shape == (obs_dim,)

        super().__init__()

        self.obs_noise_scale = PyroParam(obs_noise_scale_init,
                                         constraint=constraints.positive)
        self.trans_noise_scale_sq = PyroParam(torch.ones(state_dim),
                                              constraint=constraints.positive)
        self.trans_matrix = nn.Parameter(torch.eye(state_dim) + 0.03 * torch.randn(state_dim, state_dim))
        self.obs_matrix = nn.Parameter(0.3 * torch.randn(state_dim, obs_dim))
        self.init_noise_scale_sq = PyroParam(torch.ones(state_dim),
                                             constraint=constraints.positive)

        if learnable_observation_loc:
            self.obs_loc = nn.Parameter(torch.zeros(obs_dim))
        else:
            self.register_buffer('obs_loc', torch.zeros(obs_dim))

    def _get_init_dist(self):
        loc = self.obs_matrix.new_zeros(self.state_dim)
        return MultivariateNormal(loc, self.init_noise_scale_sq.diag_embed())

    def _get_obs_dist(self):
        return dist.Normal(self.obs_loc, self.obs_noise_scale).to_event(1)

    def _get_trans_dist(self):
        loc = self.obs_matrix.new_zeros(self.state_dim)
        return MultivariateNormal(loc, self.trans_noise_scale_sq.diag_embed())

    def get_dist(self, duration=None):
        """
        Get the :class:`~pyro.distributions.GaussianHMM` distribution that corresponds to :class:`GenericLGSSM`.

        :param int duration: Optional size of the time axis ``event_shape[0]``.
            This is required when sampling from homogeneous HMMs whose parameters
            are not expanded along the time axis.
        """
        return dist.GaussianHMM(self._get_init_dist(), self.trans_matrix, self._get_trans_dist(),
                                self.obs_matrix, self._get_obs_dist(), duration=duration)

    @pyro_method
    def log_prob(self, targets):
        """
        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape ``(T, obs_dim)``, where ``T`` is the length of the time series and ``obs_dim``
            is the dimension of the real-valued ``targets`` at each time step
        :returns torch.Tensor: A (scalar) log probability.
        """
        assert targets.dim() == 2 and targets.size(-1) == self.obs_dim
        return self.get_dist().log_prob(targets)

    @torch.no_grad()
    def _filter(self, targets):
        """
        Return the filtering state for the associated state space model.
        """
        assert targets.dim() == 2 and targets.size(-1) == self.obs_dim
        return self.get_dist().filter(targets)

    @torch.no_grad()
    def _forecast(self, N_timesteps, filtering_state, include_observation_noise=True):
        """
        Internal helper for forecasting.
        """
        N_trans_matrix = repeated_matmul(self.trans_matrix, N_timesteps)
        N_trans_obs = torch.matmul(N_trans_matrix, self.obs_matrix)
        predicted_mean = torch.matmul(filtering_state.loc, N_trans_obs)

        # first compute the contribution from filtering_state.covariance_matrix
        predicted_covar1 = torch.matmul(N_trans_obs.transpose(-1, -2),
                                        torch.matmul(filtering_state.covariance_matrix,
                                        N_trans_obs))  # N O O

        # next compute the contribution from process noise that is injected at each timestep.
        # (we need to do a cumulative sum to integrate across time)
        process_covar = self._get_trans_dist().covariance_matrix
        N_trans_obs_shift = torch.cat([self.obs_matrix.unsqueeze(0), N_trans_obs[:-1]])
        predicted_covar2 = torch.matmul(N_trans_obs_shift.transpose(-1, -2),
                                        torch.matmul(process_covar, N_trans_obs_shift))  # N O O

        predicted_covar = predicted_covar1 + torch.cumsum(predicted_covar2, dim=0)

        if include_observation_noise:
            predicted_covar = predicted_covar + self.obs_noise_scale.pow(2.0).diag_embed()

        return predicted_mean, predicted_covar

    @pyro_method
    def forecast(self, targets, N_timesteps):
        """
        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape ``(T, obs_dim)``, where ``T`` is the length of the time series and ``obs_dim``
            is the dimension of the real-valued targets at each time step. These
            represent the training data that are conditioned on for the purpose of making
            forecasts.
        :param int N_timesteps: The number of timesteps to forecast into the future from
            the final target ``targets[-1]``.
        :returns torch.distributions.MultivariateNormal: Returns a predictive MultivariateNormal distribution
            with batch shape ``(N_timesteps,)`` and event shape ``(obs_dim,)``
        """
        filtering_state = self._filter(targets)
        predicted_mean, predicted_covar = self._forecast(N_timesteps, filtering_state)
        return torch.distributions.MultivariateNormal(predicted_mean, predicted_covar)
