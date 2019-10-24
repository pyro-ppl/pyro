import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

import pyro.distributions as dist
from pyro.contrib.timeseries.base import TimeSeriesModel


class GenericLGSSM(TimeSeriesModel):
    """
    A generic Linear Gaussian State Space Model parameterized with arbitrary transition
    and observation dynamics. The targets are (implicitly) assumed to be evenly spaced in
    time. Training and inference are logarithmic in the length of the time series T.

    :param int state_dim: The dimension of latent state at each time step.
    :param int obs_dim: The dimension of the targets at each time step.
    """
    def __init__(self, obs_dim=1, log_obs_noise_scale_init=None):
        self.obs_dim = obs_dim

        if log_obs_noise_scale_init is None:
            log_obs_noise_scale_init = -2.0 * torch.ones(obs_dim)
        assert log_obs_noise_scale_init.shape == (obs_dim,)

        super(GenericLGSSM, self).__init__()

        self.log_obs_noise_scale = nn.Parameter(log_obs_noise_scale_init)
        self.log_trans_noise_scale_sq = nn.Parameter(torch.zeros(state_dim))
        self.trans_matrix = nn.Parameter(torch.eye(state_dim) + 0.03 * torch.randn(state_dim, state_dim))
        self.obs_matrix = nn.Parameter(0.3 * torch.randn(state_dim, obs_dim))
        self.log_init_noise_scale_sq = nn.Parameter(torch.zeros(state_dim))

    def _get_obs_noise_scale(self):
        return self.log_obs_noise_scale.exp()

    def _get_init_dist(self):
        loc = torch.zeros(self.state_dim, device=self.obs_matrix.device, dtype=self.obs_matrix.device)
        eye = torch.eye(self.state_dim, device=self.obs_matrix.device, dtype=self.obs_matrix.device)
        return MultivariateNormal(loc, self.log_trans_noise_scale_sq.exp() * eye)

    def _get_obs_dist(self):
        loc = torch.zeros(self.obs_dim, device=self.obs_matrix.device, dtype=self.obs_matrix.device)
        return dist.Normal(loc, self._get_obs_noise_scale()).to_event(1)

    def _get_trans_dist(self):
        loc = torch.zeros(self.state_dim, device=self.obs_matrix.device, dtype=self.obs_matrix.device)
        eye = torch.eye(self.state_dim, device=self.obs_matrix.device, dtype=self.obs_matrix.device)
        return MultivariateNormal(loc, self.log_trans_noise_scale.exp() * eye)

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

    def _forecast(self, dts, filtering_state, include_observation_noise=True):
        """
        Internal helper for forecasting.
        """
        assert dts.dim() == 1
        dts = dts.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        predicted_mean = torch.matmul(filtering_state.loc.unsqueeze(-2), trans_matrix).squeeze(-2)[..., 0]
        predicted_function_covar = torch.matmul(trans_matrix.transpose(-1, -2),
                                                torch.matmul(filtering_state.covariance_matrix,
                                                trans_matrix))[..., 0, 0] + process_covar[..., 0, 0]

        if include_observation_noise:
            predicted_function_covar = predicted_function_covar + self._get_obs_noise_scale().pow(2.0)
        return predicted_mean, predicted_function_covar

    def forecast(self, targets, dts):
        """
        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape `(T, obs_dim)`, where `T` is the length of the time series and `obs_dim`
            is the dimension of the real-valued targets at each time step. These
            represent the training data that are conditioned on for the purpose of making
            forecasts.
        :param torch.Tensor dts: A 1-dimensional tensor of times to forecast into the future,
            with zero corresponding to the time of the final target `targets[-1]`.
        :returns torch.distributions.Normal: Returns a predictive Normal distribution with batch shape `(S,)` and
            event shape `(obs_dim,)`, where `S` is the size of `dts`.
        """
        filtering_state = self._filter(targets)
        predicted_mean, predicted_covar = self._forecast(dts, filtering_state)
        return torch.distributions.Normal(predicted_mean, predicted_covar.sqrt())
