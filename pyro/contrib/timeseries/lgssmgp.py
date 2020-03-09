# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, constraints

import pyro.distributions as dist
from pyro.contrib.timeseries.base import TimeSeriesModel
from pyro.nn import PyroParam, pyro_method
from pyro.ops.ssm_gp import MaternKernel
from pyro.ops.tensor_utils import block_diag_embed, repeated_matmul


class GenericLGSSMWithGPNoiseModel(TimeSeriesModel):
    """
    A generic Linear Gaussian State Space Model parameterized with arbitrary time invariant
    transition and observation dynamics together with separate Gaussian Process noise models
    for each output dimension. In more detail, the generative process is:

        :math:`y_i(t) = \\sum_j A_{ij} z_j(t) + f_i(t) + \\epsilon_i(t)`

    where the latent variables :math:`{\\bf z}(t)` follow generic time invariant Linear Gaussian dynamics
    and the :math:`f_i(t)` are Gaussian Processes with Matern kernels.

    The targets are (implicitly) assumed to be evenly spaced in time. In particular a timestep of
    :math:`dt=1.0` for the continuous-time GP dynamics corresponds to a single discrete step of
    the :math:`{\\bf z}`-space dynamics. Training and inference are logarithmic in the length of
    the time series T.

    :param int obs_dim: The dimension of the targets at each time step.
    :param int state_dim: The dimension of the :math:`{\\bf z}` latent state at each time step.
    :param float nu: The order of the Matern kernel; one of 0.5, 1.5 or 2.5.
    :param torch.Tensor length_scale_init: optional initial values for the kernel length scale
        given as a ``obs_dim``-dimensional tensor
    :param torch.Tensor kernel_scale_init: optional initial values for the kernel scale
        given as a ``obs_dim``-dimensional tensor
    :param torch.Tensor obs_noise_scale_init: optional initial values for the observation noise scale
        given as a ``obs_dim``-dimensional tensor
    :param bool learnable_observation_loc: whether the mean of the observation model should be learned or not;
            defaults to False.
    """
    def __init__(self, obs_dim=1, state_dim=2, nu=1.5, obs_noise_scale_init=None,
                 length_scale_init=None, kernel_scale_init=None,
                 learnable_observation_loc=False):
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.nu = nu

        if obs_noise_scale_init is None:
            obs_noise_scale_init = 0.2 * torch.ones(obs_dim)
        assert obs_noise_scale_init.shape == (obs_dim,)

        super().__init__()

        self.kernel = MaternKernel(nu=nu, num_gps=obs_dim,
                                   length_scale_init=length_scale_init,
                                   kernel_scale_init=kernel_scale_init)
        self.dt = 1.0
        self.full_state_dim = self.kernel.state_dim * obs_dim + state_dim
        self.full_gp_state_dim = self.kernel.state_dim * obs_dim

        self.obs_noise_scale = PyroParam(obs_noise_scale_init,
                                         constraint=constraints.positive)
        self.trans_noise_scale_sq = PyroParam(torch.ones(state_dim),
                                              constraint=constraints.positive)
        self.z_trans_matrix = nn.Parameter(torch.eye(state_dim) + 0.03 * torch.randn(state_dim, state_dim))
        self.z_obs_matrix = nn.Parameter(0.3 * torch.randn(state_dim, obs_dim))
        self.init_noise_scale_sq = PyroParam(torch.ones(state_dim),
                                             constraint=constraints.positive)

        gp_obs_matrix = torch.zeros(self.kernel.state_dim * obs_dim, obs_dim)
        for i in range(obs_dim):
            gp_obs_matrix[self.kernel.state_dim * i, i] = 1.0
        self.register_buffer("gp_obs_matrix", gp_obs_matrix)

        self.obs_selector = torch.tensor([self.kernel.state_dim * d for d in range(obs_dim)], dtype=torch.long)

        if learnable_observation_loc:
            self.obs_loc = nn.Parameter(torch.zeros(obs_dim))
        else:
            self.register_buffer('obs_loc', torch.zeros(obs_dim))

    def _get_obs_matrix(self):
        # (obs_dim + state_dim, obs_dim) => (gp_state_dim * obs_dim + state_dim, obs_dim)
        return torch.cat([self.gp_obs_matrix, self.z_obs_matrix], dim=0)

    def _get_init_dist(self):
        loc = self.z_trans_matrix.new_zeros(self.full_state_dim)
        covar = self.z_trans_matrix.new_zeros(self.full_state_dim, self.full_state_dim)
        covar[:self.full_gp_state_dim, :self.full_gp_state_dim] = block_diag_embed(self.kernel.stationary_covariance())
        covar[self.full_gp_state_dim:, self.full_gp_state_dim:] = self.init_noise_scale_sq.diag_embed()
        return MultivariateNormal(loc, covar)

    def _get_obs_dist(self):
        return dist.Normal(self.obs_loc, self.obs_noise_scale).to_event(1)

    def get_dist(self, duration=None):
        """
        Get the :class:`~pyro.distributions.GaussianHMM` distribution that corresponds
        to :class:`GenericLGSSMWithGPNoiseModel`.

        :param int duration: Optional size of the time axis ``event_shape[0]``.
            This is required when sampling from homogeneous HMMs whose parameters
            are not expanded along the time axis.
        """
        gp_trans_matrix, gp_process_covar = self.kernel.transition_matrix_and_covariance(dt=self.dt)

        trans_covar = self.z_trans_matrix.new_zeros(self.full_state_dim, self.full_state_dim)
        trans_covar[:self.full_gp_state_dim, :self.full_gp_state_dim] = block_diag_embed(gp_process_covar)
        trans_covar[self.full_gp_state_dim:, self.full_gp_state_dim:] = self.trans_noise_scale_sq.diag_embed()
        trans_dist = MultivariateNormal(trans_covar.new_zeros(self.full_state_dim), trans_covar)

        full_trans_mat = trans_covar.new_zeros(self.full_state_dim, self.full_state_dim)
        full_trans_mat[:self.full_gp_state_dim, :self.full_gp_state_dim] = block_diag_embed(gp_trans_matrix)
        full_trans_mat[self.full_gp_state_dim:, self.full_gp_state_dim:] = self.z_trans_matrix

        return dist.GaussianHMM(self._get_init_dist(), full_trans_mat, trans_dist,
                                self._get_obs_matrix(), self._get_obs_dist(), duration=duration)

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
        dts = torch.arange(N_timesteps, dtype=self.z_trans_matrix.dtype, device=self.z_trans_matrix.device) + 1.0
        dts = dts.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        gp_trans_matrix, gp_process_covar = self.kernel.transition_matrix_and_covariance(dt=dts)
        gp_trans_matrix = block_diag_embed(gp_trans_matrix)
        gp_process_covar = block_diag_embed(gp_process_covar[..., 0:1, 0:1])

        N_trans_matrix = repeated_matmul(self.z_trans_matrix, N_timesteps)
        N_trans_obs = torch.matmul(N_trans_matrix, self.z_obs_matrix)

        # z-state contribution + gp contribution
        predicted_mean1 = torch.matmul(filtering_state.loc[-self.state_dim:].unsqueeze(-2), N_trans_obs).squeeze(-2)
        predicted_mean2 = torch.matmul(filtering_state.loc[:self.full_gp_state_dim].unsqueeze(-2),
                                       gp_trans_matrix[..., self.obs_selector]).squeeze(-2)
        predicted_mean = predicted_mean1 + predicted_mean2

        # first compute the contributions from filtering_state.covariance_matrix: z-space and gp
        fs_cov = filtering_state.covariance_matrix
        predicted_covar1z = torch.matmul(N_trans_obs.transpose(-1, -2),
                                         torch.matmul(fs_cov[self.full_gp_state_dim:, self.full_gp_state_dim:],
                                         N_trans_obs))  # N O O
        gp_trans = gp_trans_matrix[..., self.obs_selector]
        predicted_covar1gp = torch.matmul(gp_trans.transpose(-1, -2),
                                          torch.matmul(fs_cov[:self.full_gp_state_dim:, :self.full_gp_state_dim],
                                          gp_trans))

        # next compute the contribution from process noise that is injected at each timestep.
        # (we need to do a cumulative sum to integrate across time for the z-state contribution)
        z_process_covar = self.trans_noise_scale_sq.diag_embed()
        N_trans_obs_shift = torch.cat([self.z_obs_matrix.unsqueeze(0), N_trans_obs[0:-1]])
        predicted_covar2z = torch.matmul(N_trans_obs_shift.transpose(-1, -2),
                                         torch.matmul(z_process_covar, N_trans_obs_shift))  # N O O

        predicted_covar = predicted_covar1z + predicted_covar1gp + gp_process_covar + \
            torch.cumsum(predicted_covar2z, dim=0)

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
        return MultivariateNormal(predicted_mean, predicted_covar)
