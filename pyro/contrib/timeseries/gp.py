import math

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

import pyro.distributions as dist
from pyro.ops.ssm_gp import MaternKernel
from pyro.contrib.timeseries.base import TimeSeriesModel
from pyro.ops.tensor_utils import block_diag_embed


class IndependentMaternGP(TimeSeriesModel):
    """
    A time series model in which each output dimension is modeled independently
    with a univariate Gaussian Process with a Matern kernel. The targets are assumed
    to be evenly spaced in time. Training and inference are logarithmic in the length
    of the time series T.

    :param float nu: The order of the Matern kernel; one of 0.5, 1.5 or 2.5.
    :param float dt: The time spacing between neighboring observations of the time series.
    :param int obs_dim: The dimension of the targets at each time step.
    :param torch.Tensor log_length_scale_init: optional initial values for the kernel length scale
        given as a `obs_dim`-dimensional tensor
    :param torch.Tensor log_kernel_scale_init: optional initial values for the kernel scale
        given as a `obs_dim`-dimensional tensor
    :param torch.Tensor log_obs_noise_scale_init: optional initial values for the observation noise scale
        given as a `obs_dim`-dimensional tensor
    """
    def __init__(self, nu=1.5, dt=1.0, obs_dim=1,
                 log_length_scale_init=None, log_kernel_scale_init=None,
                 log_obs_noise_scale_init=None):
        self.nu = nu
        self.dt = dt
        self.obs_dim = obs_dim

        if log_obs_noise_scale_init is None:
            log_obs_noise_scale_init = -2.0 * torch.ones(obs_dim)
        assert log_obs_noise_scale_init.shape == (obs_dim,)

        super().__init__()

        self.kernel = MaternKernel(nu=nu, num_gps=obs_dim,
                                   log_length_scale_init=log_length_scale_init,
                                   log_kernel_scale_init=log_kernel_scale_init)

        self.log_obs_noise_scale = nn.Parameter(log_obs_noise_scale_init)

        obs_matrix = [1.0] + [0.0] * (self.kernel.state_dim - 1)
        self.register_buffer("obs_matrix", torch.tensor(obs_matrix).unsqueeze(-1))

    def _get_obs_noise_scale(self):
        return self.log_obs_noise_scale.exp()

    def _get_init_dist(self):
        return torch.distributions.MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, self.kernel.state_dim),
                                                      self.kernel.stationary_covariance().squeeze(-3))

    def _get_obs_dist(self):
        return dist.Normal(self.obs_matrix.new_zeros(self.obs_dim, 1, 1),
                           self._get_obs_noise_scale().unsqueeze(-1).unsqueeze(-1)).to_event(1)

    def _get_dist(self):
        """
        Get the `GaussianHMM` distribution that corresponds to `obs_dim`-many independent Matern GPs.
        """
        trans_matrix, process_covar = self.kernel.transition_matrix_and_covariance(dt=self.dt)
        trans_dist = MultivariateNormal(self.obs_matrix.new_zeros(self.obs_dim, 1, self.kernel.state_dim),
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

    @torch.no_grad()
    def _filter(self, targets):
        """
        Return the filtering state for the associated state space model.
        """
        assert targets.dim() == 2 and targets.size(-1) == self.obs_dim
        return self._get_dist().filter(targets.t().unsqueeze(-1))

    @torch.no_grad()
    def _forecast(self, dts, filtering_state, include_observation_noise=True):
        """
        Internal helper for forecasting.
        """
        assert dts.dim() == 1
        dts = dts.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        trans_matrix, process_covar = self.kernel.transition_matrix_and_covariance(dt=dts)
        predicted_mean = torch.matmul(filtering_state.loc.unsqueeze(-2), trans_matrix).squeeze(-2)[..., 0]
        predicted_function_covar = torch.matmul(trans_matrix.transpose(-1, -2), torch.matmul(
                                                filtering_state.covariance_matrix, trans_matrix))[..., 0, 0] + \
            process_covar[..., 0, 0]

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


class LinearlyCoupledMaternGP(TimeSeriesModel):
    """
    A time series model in which each output dimension is modeled as a linear combination
    of shared univariate Gaussian Processes with Matern kernels.

    In more detail, the generative process is:

        :math:`y_i(t) = \\sum_j A_{ij} f_j(t) + \\epsilon_i`

    The targets :math:`y_i` are assumed to be evenly spaced in time. Training and inference
    are logarithmic in the length of the time series T.

    :param float nu: The order of the Matern kernel; one of 0.5, 1.5 or 2.5.
    :param float dt: The time spacing between neighboring observations of the time series.
    :param int obs_dim: The dimension of the targets at each time step.
    :param int num_gps: The number of independent GPs that are mixed to model the time series.
        Typical values might be :math:`\\N_{\\rm gp} \\in [\\D_{\\rm obs} / 2, \\D_{\\rm obs}]`
    :param torch.Tensor log_length_scale_init: optional initial values for the kernel length scale
        given as a `num_gps`-dimensional tensor
    :param torch.Tensor log_kernel_scale_init: optional initial values for the kernel scale
        given as a `num_gps`-dimensional tensor
    :param torch.Tensor log_obs_noise_scale_init: optional initial values for the observation noise scale
        given as a `obs_dim`-dimensional tensor
    """
    def __init__(self, nu=1.5, dt=1.0, obs_dim=2, num_gps=1,
                 log_length_scale_init=None, log_kernel_scale_init=None,
                 log_obs_noise_scale_init=None):
        self.nu = nu
        self.dt = dt
        assert obs_dim > 1, "If obs_dim==1 you should use IndependentMaternGP"
        self.obs_dim = obs_dim
        self.num_gps = num_gps

        if log_obs_noise_scale_init is None:
            log_obs_noise_scale_init = -2.0 * torch.ones(obs_dim)
        assert log_obs_noise_scale_init.shape == (obs_dim,)

        self.dt = dt
        self.obs_dim = obs_dim
        self.num_gps = num_gps

        super().__init__()

        self.kernel = MaternKernel(nu=nu, num_gps=num_gps,
                                   log_length_scale_init=log_length_scale_init,
                                   log_kernel_scale_init=log_kernel_scale_init)
        self.full_state_dim = num_gps * self.kernel.state_dim

        self.log_obs_noise_scale = nn.Parameter(log_obs_noise_scale_init)
        self.A = nn.Parameter(0.3 * torch.randn(self.num_gps, self.obs_dim))

    def _get_obs_matrix(self):
        # (num_gps, obs_dim) => (state_dim * num_gps, obs_dim)
        return self.A.repeat_interleave(self.kernel.state_dim, dim=0) * \
            self.A.new_tensor([1.0] + [0.0] * (self.kernel.state_dim - 1)).repeat(self.num_gps).unsqueeze(-1)

    def _get_obs_noise_scale(self):
        return self.log_obs_noise_scale.exp()

    def _stationary_covariance(self):
        return block_diag_embed(self.kernel.stationary_covariance())

    def _get_init_dist(self):
        loc = self.A.new_zeros(self.full_state_dim)
        return MultivariateNormal(loc, self._stationary_covariance())

    def _get_obs_dist(self):
        loc = self.A.new_zeros(self.obs_dim)
        return dist.Normal(loc, self._get_obs_noise_scale()).to_event(1)

    def _get_dist(self):
        """
        Get the `GaussianHMM` distribution that corresponds to a `LinearlyCoupledMaternGP`.
        """
        trans_matrix, process_covar = self.kernel.transition_matrix_and_covariance(dt=self.dt)
        trans_matrix = block_diag_embed(trans_matrix)
        process_covar = block_diag_embed(process_covar)
        loc = self.A.new_zeros(self.full_state_dim)
        trans_dist = MultivariateNormal(loc, process_covar)
        return dist.GaussianHMM(self._get_init_dist(), trans_matrix, trans_dist,
                                self._get_obs_matrix(), self._get_obs_dist())

    def log_prob(self, targets):
        """
        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape `(T, obs_dim)`, where `T` is the length of the time series and `obs_dim`
            is the dimension of the real-valued `targets` at each time step
        :returns torch.Tensor: a (scalar) log probability
        """
        assert targets.dim() == 2 and targets.size(-1) == self.obs_dim
        return self._get_dist().log_prob(targets)

    @torch.no_grad()
    def _filter(self, targets):
        """
        Return the filtering state for the associated state space model.
        """
        assert targets.dim() == 2 and targets.size(-1) == self.obs_dim
        return self._get_dist().filter(targets)

    @torch.no_grad()
    def _forecast(self, dts, filtering_state, include_observation_noise=True, full_covar=True):
        """
        Internal helper for forecasting.
        """
        assert dts.dim() == 1
        dts = dts.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        trans_mat, process_covar = self.kernel.transition_matrix_and_covariance(dt=dts)
        trans_mat = block_diag_embed(trans_mat)  # S x full_state_dim x full_state_dim
        process_covar = block_diag_embed(process_covar)  # S x full_state_dim x full_state_dim
        obs_matrix = self._get_obs_matrix()  # full_state_dim x obs_dim
        trans_obs = torch.matmul(trans_mat, obs_matrix)  # S x full_state_dim x obs_dim
        predicted_mean = torch.matmul(filtering_state.loc.unsqueeze(-2), trans_obs).squeeze(-2)
        predicted_function_covar = torch.matmul(trans_obs.transpose(-1, -2),
                                                torch.matmul(filtering_state.covariance_matrix,
                                                trans_obs))
        predicted_function_covar = predicted_function_covar + \
            torch.matmul(obs_matrix.transpose(-1, -2), torch.matmul(process_covar, obs_matrix))

        if include_observation_noise:
            eye = torch.eye(self.obs_dim, dtype=self.A.dtype, device=self.A.device)
            obs_noise = self._get_obs_noise_scale().pow(2.0) * eye
            predicted_function_covar = predicted_function_covar + obs_noise
        if not full_covar:
            predicted_function_covar = predicted_function_covar.diagonal(dim1=-1, dim2=-2)

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
        :returns torch.distributions.MultivariateNormal: Returns a predictive MultivariateNormal
            distribution with batch shape `(S,)` and event shape `(obs_dim,)`,
            where `S` is the size of `dts`.
        """
        filtering_state = self._filter(targets)
        predicted_mean, predicted_covar = self._forecast(dts, filtering_state)
        return MultivariateNormal(predicted_mean, predicted_covar)


class DependentMaternGP(TimeSeriesModel):
    """
    A time series model in which each output dimension is modeled as a univariate Gaussian Process
    with a Matern kernel. The different output dimensions become correlated because the Gaussian
    Processes are driven by a correlated Wiener process; see reference [1] for details.
    The targets are assumed to be evenly spaced in time. Training and inference are logarithmic
    in the length of the time series T.

    :param float nu: The order of the Matern kernel; must be 1.5.
    :param float dt: The time spacing between neighboring observations of the time series.
    :param int obs_dim: The dimension of the targets at each time step.
    :param torch.Tensor log_length_scale_init: optional initial values for the kernel length scale
        given as a `obs_dim`-dimensional tensor
    :param torch.Tensor log_obs_noise_scale_init: optional initial values for the observation noise scale
        given as a `obs_dim`-dimensional tensor

    References
    [1] "Dependent Matern Processes for Multivariate Time Series," Alexander Vandenberg-Rodes, Babak Shahbaba.
    """
    def __init__(self, nu=1.5, dt=1.0, obs_dim=1,
                 log_length_scale_init=None, log_obs_noise_scale_init=None):

        if nu != 1.5:
            raise NotImplementedError("The only supported value of nu is 1.5")

        self.dt = dt
        self.obs_dim = obs_dim

        if log_obs_noise_scale_init is None:
            log_obs_noise_scale_init = -2.0 * torch.ones(obs_dim)
        assert log_obs_noise_scale_init.shape == (obs_dim,)

        super().__init__()

        self.kernel = MaternKernel(nu=nu, num_gps=obs_dim,
                                   log_length_scale_init=log_length_scale_init)
        self.full_state_dim = self.kernel.state_dim * obs_dim

        # we demote self.kernel.log_kernel_scale from being a nn.Parameter
        # since the relevant scales are now encoded in the wiener noise matrix
        del self.kernel.log_kernel_scale
        self.kernel.register_buffer("log_kernel_scale", torch.zeros(obs_dim))

        self.log_obs_noise_scale = nn.Parameter(log_obs_noise_scale_init)
        self.log_diag_wiener_noise = nn.Parameter(torch.zeros(obs_dim))
        self.off_diag_wiener_noise = nn.Parameter(0.03 * torch.randn(obs_dim, obs_dim))

        obs_matrix = torch.zeros(self.full_state_dim, obs_dim)
        for i in range(obs_dim):
            obs_matrix[self.kernel.state_dim * i, i] = 1.0
        self.register_buffer("obs_matrix", obs_matrix)

    def _get_obs_noise_scale(self):
        return self.log_obs_noise_scale.exp()

    def _get_init_dist(self, stationary_covariance):
        return torch.distributions.MultivariateNormal(self.obs_matrix.new_zeros(self.full_state_dim),
                                                      stationary_covariance)

    def _get_obs_dist(self):
        return dist.Normal(self.obs_matrix.new_zeros(self.obs_dim),
                           self._get_obs_noise_scale()).to_event(1)

    def _get_wiener_cov(self):
        eye = torch.eye(self.obs_dim, device=self.obs_matrix.device, dtype=self.obs_matrix.dtype)
        chol = eye * self.log_diag_wiener_noise.exp() + self.off_diag_wiener_noise.tril(-1)
        wiener_cov = torch.mm(chol, chol.t()).reshape(self.obs_dim, 1, self.obs_dim, 1)
        wiener_cov = wiener_cov * wiener_cov.new_ones(self.kernel.state_dim, 1, self.kernel.state_dim)
        return wiener_cov.reshape(self.full_state_dim, self.full_state_dim)

    def _stationary_covariance(self):
        rho_j = math.sqrt(3.0) * (-self.kernel.log_length_scale).exp().unsqueeze(-1).unsqueeze(-1)
        rho_i = rho_j.unsqueeze(-1)
        block = 2.0 * self.kernel.mask00 + \
            (rho_i - rho_j) * (self.kernel.mask01 - self.kernel.mask10) + \
            (2.0 * rho_i * rho_j) * self.kernel.mask11
        block = block / (rho_i + rho_j).pow(3.0)
        block = block.transpose(-2, -3).reshape(self.full_state_dim, self.full_state_dim)
        return self._get_wiener_cov() * block

    def _get_trans_dist(self, trans_matrix, stationary_covariance):
        covar = stationary_covariance - torch.matmul(trans_matrix.transpose(-1, -2),
                                                     torch.matmul(stationary_covariance, trans_matrix))
        return MultivariateNormal(covar.new_zeros(self.full_state_dim), covar)

    def _trans_matrix_distribution_stat_covar(self, dts):
        stationary_covariance = self._stationary_covariance()
        trans_matrix = self.kernel.transition_matrix(dt=dts)
        trans_matrix = block_diag_embed(trans_matrix)
        trans_dist = self._get_trans_dist(trans_matrix, stationary_covariance)
        return trans_matrix, trans_dist, stationary_covariance

    def _get_dist(self):
        """
        Get the `GaussianHMM` distribution that corresponds to a `DependentMaternGP`
        """
        trans_matrix, trans_dist, stat_covar = self._trans_matrix_distribution_stat_covar(self.dt)
        return dist.GaussianHMM(self._get_init_dist(stat_covar), trans_matrix,
                                trans_dist, self.obs_matrix, self._get_obs_dist())

    def log_prob(self, targets):
        """
        :param torch.Tensor targets: A 2-dimensional tensor of real-valued targets
            of shape `(T, obs_dim)`, where `T` is the length of the time series and `obs_dim`
            is the dimension of the real-valued `targets` at each time step
        :returns torch.Tensor: A (scalar) log probability
        """
        assert targets.dim() == 2 and targets.size(-1) == self.obs_dim
        return self._get_dist().log_prob(targets)

    @torch.no_grad()
    def _filter(self, targets):
        """
        Return the filtering state for the associated state space model.
        """
        assert targets.dim() == 2 and targets.size(-1) == self.obs_dim
        return self._get_dist().filter(targets)

    @torch.no_grad()
    def _forecast(self, dts, filtering_state, include_observation_noise=True):
        """
        Internal helper for forecasting.
        """
        assert dts.dim() == 1
        dts = dts.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        trans_matrix, trans_dist, _ = self._trans_matrix_distribution_stat_covar(dts)
        trans_obs = torch.matmul(trans_matrix, self.obs_matrix)

        predicted_mean = torch.matmul(filtering_state.loc.unsqueeze(-2), trans_obs).squeeze(-2)
        predicted_function_covar = torch.matmul(trans_obs.transpose(-1, -2),
                                                torch.matmul(filtering_state.covariance_matrix, trans_obs)) + \
            torch.matmul(self.obs_matrix.t(), torch.matmul(trans_dist.covariance_matrix, self.obs_matrix))

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
        :returns torch.distributions.MultivariateNormal: Returns a predictive MultivariateNormal
            distribution with batch shape `(S,)` and event shape `(obs_dim,)`, where `S` is the size of `dts`.
        """
        filtering_state = self._filter(targets)
        predicted_mean, predicted_covar = self._forecast(dts, filtering_state)
        return MultivariateNormal(predicted_mean, predicted_covar)
