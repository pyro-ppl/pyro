import torch
import torch.nn as nn


class MaternKernel(nn.Module):
    """
    Provides the building blocks for representing univariate Gaussian Processes (GPs)
    with Matern kernels as state space models.

    :param float nu: The order of the Matern kernel (1.5 or 2.5)
    :param int num_gps: the number of GPs
    :param torch.tensor log_length_scale_init: optional `num_gps`-dimensional vector of initializers for the length scale
    :param torch.tensor log_kernel_scale_init: optional `num_gps`-dimensional vector of initializers for the kernel scale

    **References**

    [1] `Kalman Filtering and Smoothing Solutions to Temporal Gaussian Process Regression Models`,
    Jouni Hartikainen and Simo Sarkka.
    """
    def __init__(self, nu=1.5, num_gps=1, log_length_scale_init=None, log_kernel_scale=None):
        assert nu in [1.5, 2.5], "The only supported values of nu are 1.5 and 2.5"
        self.nu = nu
        self.state_dim = {1.5: 2, 2.5: 3}[nu]

        if log_length_scale_init is not None:
            assert log_length_scale_init.shape == (num_gps,)
        else:
            log_length_scale_init = 0.01 * torch.randn(num_gps)

        if log_kernel_scale_init is not None:
            assert log_kernel_scale_init.shape == (num_gps,)
        else:
            log_kernel_scale_init = 0.01 * torch.randn(num_gps)

        super(MaternKernel, self).__init__()

        self.log_length_scale = nn.Parameter(log_length_scale_init)
        self.log_kernel_scale = nn.Parameter(log_kernel_scale_init)

        for x in range(self.state_dim):
            for y in range(self.state_dim):
                mask = torch.zeros(self.state_dim, self.state_dim)
                mask[x, y] = 1.0
                self.register_buffer("mask{}{}".format(x, y), mask)

    def transition_matrix(self, dt):
    """
    Compute the (exponentiated) transition matrix of the GP latent space.
    The resulting matrix has shape (num_gps, old_state, new_state), i.e. this
    matrix multiplied states from the right.

    See section 5 in reference [1] for details.

    :param float dt: the time interval over which the GP latent space evolves.
    :returns torch.Tensor: a 3-dimensional tensor of transition matrices of shape
        (num_gps, old_state, new_state).
    """
    if self.nu == 1.5:
	rho = self.log_length_scale.exp().unsqueeze(-1).unsqueeze(-1)
	dt_rho = dt / rho
	trans = (1.0 + root_three * dt_rho) * self.mask00 + \
		(-3.0 * dt_rho / rho) * self.mask01 + \
		dt * self.mask10 + \
		(1.0 - root_three * dt_rho) * self.mask11
	return torch.exp(-root_three * dt_rho) * trans
    else:
        raise NotImplementedError
