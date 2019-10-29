import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

import pyro.distributions as dist
from pyro.contrib.timeseries.base import TimeSeriesModel
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
    def __init__(self, obs_dim=1, state_dim=2, log_obs_noise_scale_init=None,
                 learnable_observation_loc=False):
        self.obs_dim = obs_dim
        self.state_dim = state_dim

        if log_obs_noise_scale_init is None:
            log_obs_noise_scale_init = -2.0 * torch.ones(obs_dim)
        assert log_obs_noise_scale_init.shape == (obs_dim,)

        super(GenericLGSSM, self).__init__()

        self.log_obs_noise_scale = nn.Parameter(log_obs_noise_scale_init)
        self.log_trans_noise_scale_sq = nn.Parameter(torch.zeros(state_dim))
        self.trans_matrix = nn.Parameter(torch.eye(state_dim) + 0.03 * torch.randn(state_dim, state_dim))
        self.obs_matrix = nn.Parameter(0.3 * torch.randn(state_dim, obs_dim))
        self.log_init_noise_scale_sq = nn.Parameter(torch.zeros(state_dim))

        if learnable_observation_loc:
            self.obs_loc = nn.Parameter(torch.zeros(obs_dim))
        else:
            self.register_buffer('obs_loc', torch.zeros(obs_dim))

    def _get_obs_noise_scale(self):
        return self.log_obs_noise_scale.exp()

    def _get_init_dist(self):
        loc = self.obs_matrix.new_zeros(self.state_dim)
        eye = torch.eye(self.state_dim, device=loc.device, dtype=loc.dtype)
        return MultivariateNormal(loc, self.log_init_noise_scale_sq.exp() * eye)

    def _get_obs_dist(self):
        return dist.Normal(self.obs_loc, self._get_obs_noise_scale()).to_event(1)

    def _get_trans_dist(self):
        loc = self.obs_matrix.new_zeros(self.state_dim)
        eye = torch.eye(self.state_dim, device=loc.device, dtype=loc.dtype)
        return MultivariateNormal(loc, self.log_trans_noise_scale_sq.exp() * eye)

    def _get_dist(self):
        """
        Get the `GaussianHMM` distribution that corresponds to `GenericLGSSM`.
        """
        return dist.GaussianHMM(self._get_init_dist(), self.trans_matrix, self._get_trans_dist(),
                                self.obs_matrix, self._get_obs_dist())

    def log_prob(self, targets):
        """
        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape `(T, obs_dim)`, where `T` is the length of the time series and `obs_dim`
            is the dimension of the real-valued `targets` at each time step
        :returns torch.Tensor: A (scalar) log probability.
        """
        assert targets.dim() == 2 and targets.size(-1) == self.obs_dim
        return self._get_dist().log_prob(targets)

    def _filter(self, targets):
        """
        Return the filtering state for the associated state space model.
        """
        assert targets.dim() == 2 and targets.size(-1) == self.obs_dim
        return self._get_dist().filter(targets)

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
            eye = torch.eye(self.obs_dim, device=self.obs_matrix.device, dtype=self.obs_matrix.dtype)
            predicted_covar = predicted_covar + self._get_obs_noise_scale().pow(2.0) * eye

        return predicted_mean, predicted_covar

    def forecast(self, targets, N_timesteps):
        """
        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape `(T, obs_dim)`, where `T` is the length of the time series and `obs_dim`
            is the dimension of the real-valued targets at each time step. These
            represent the training data that are conditioned on for the purpose of making
            forecasts.
        :param int N_timesteps: The number of timesteps to forecast into the future from
            the final target `targets[-1]`.
        :returns torch.distributions.MultivariateNormal: Returns a predictive MultivariateNormal distribution
            with batch shape `(N_timesteps,)` and event shape `(obs_dim,)`
        """
        filtering_state = self._filter(targets)
        predicted_mean, predicted_covar = self._forecast(N_timesteps, filtering_state)
        return torch.distributions.MultivariateNormal(predicted_mean, predicted_covar)
