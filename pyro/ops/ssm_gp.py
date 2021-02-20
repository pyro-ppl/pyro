# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions import constraints

from pyro.nn import PyroModule, pyro_method, PyroParam

root_three = math.sqrt(3.0)
root_five = math.sqrt(5.0)
five_thirds = 5.0 / 3.0


class MaternKernel(PyroModule):
    """
    Provides the building blocks for representing univariate Gaussian Processes (GPs)
    with Matern kernels as state space models.

    :param float nu: The order of the Matern kernel (one of 0.5, 1.5 or 2.5)
    :param int num_gps: the number of GPs
    :param torch.Tensor length_scale_init: optional `num_gps`-dimensional vector of initializers
        for the length scale
    :param torch.Tensor kernel_scale_init: optional `num_gps`-dimensional vector of initializers
        for the kernel scale

    **References**

    [1] `Kalman Filtering and Smoothing Solutions to Temporal Gaussian Process Regression Models`,
        Jouni Hartikainen and Simo Sarkka.
    [2] `Stochastic Differential Equation Methods for Spatio-Temporal Gaussian Process Regression`,
        Arno Solin.
    """
    def __init__(self, nu=1.5, num_gps=1, length_scale_init=None, kernel_scale_init=None):
        if nu not in [0.5, 1.5, 2.5]:
            raise NotImplementedError("The only supported values of nu are 0.5, 1.5 and 2.5")
        self.nu = nu
        self.state_dim = {0.5: 1, 1.5: 2, 2.5: 3}[nu]
        self.num_gps = num_gps

        if length_scale_init is None:
            length_scale_init = torch.ones(num_gps)
        assert length_scale_init.shape == (num_gps,)

        if kernel_scale_init is None:
            kernel_scale_init = torch.ones(num_gps)
        assert kernel_scale_init.shape == (num_gps,)

        super().__init__()

        self.length_scale = PyroParam(length_scale_init, constraint=constraints.positive)
        self.kernel_scale = PyroParam(kernel_scale_init, constraint=constraints.positive)

        if self.state_dim > 1:
            for x in range(self.state_dim):
                for y in range(self.state_dim):
                    mask = torch.zeros(self.state_dim, self.state_dim)
                    mask[x, y] = 1.0
                    self.register_buffer("mask{}{}".format(x, y), mask)

    @pyro_method
    def transition_matrix(self, dt):
        """
        Compute the (exponentiated) transition matrix of the GP latent space.
        The resulting matrix has layout (num_gps, old_state, new_state), i.e. this
        matrix multiplies states from the right.

        See section 5 in reference [1] for details.

        :param float dt: the time interval over which the GP latent space evolves.
        :returns torch.Tensor: a 3-dimensional tensor of transition matrices of shape
            (num_gps, state_dim, state_dim).
        """
        if self.nu == 0.5:
            rho = self.length_scale.unsqueeze(-1).unsqueeze(-1)
            return torch.exp(-dt / rho)
        elif self.nu == 1.5:
            rho = self.length_scale.unsqueeze(-1).unsqueeze(-1)
            dt_rho = dt / rho
            trans = (1.0 + root_three * dt_rho) * self.mask00 + \
                (-3.0 * dt_rho / rho) * self.mask01 + \
                dt * self.mask10 + \
                (1.0 - root_three * dt_rho) * self.mask11
            return torch.exp(-root_three * dt_rho) * trans
        elif self.nu == 2.5:
            rho = self.length_scale.unsqueeze(-1).unsqueeze(-1)
            dt_rho = root_five * dt / rho
            dt_rho_sq = dt_rho.pow(2.0)
            dt_rho_cu = dt_rho.pow(3.0)
            dt_rho_qu = dt_rho.pow(4.0)
            dt_sq = dt ** 2.0
            trans = (1.0 + dt_rho + 0.5 * dt_rho_sq) * self.mask00 + \
                (-0.5 * dt_rho_cu / dt) * self.mask01 + \
                ((0.5 * dt_rho_qu - dt_rho_cu) / dt_sq) * self.mask02 + \
                ((dt_rho + 1.0) * dt) * self.mask10 + \
                (1.0 + dt_rho - dt_rho_sq) * self.mask11 + \
                ((dt_rho_cu - 3.0 * dt_rho_sq) / dt) * self.mask12 + \
                (0.5 * dt_sq) * self.mask20 + \
                ((1.0 - 0.5 * dt_rho) * dt) * self.mask21 + \
                (1.0 - 2.0 * dt_rho + 0.5 * dt_rho_sq) * self.mask22
            return torch.exp(-dt_rho) * trans

    @pyro_method
    def stationary_covariance(self):
        """
        Compute the stationary state covariance. See Eqn. 3.26 in reference [2].

        :returns torch.Tensor: a 3-dimensional tensor of covariance matrices of shape
            (num_gps, state_dim, state_dim).
        """
        if self.nu == 0.5:
            sigmasq = self.kernel_scale.pow(2).unsqueeze(-1).unsqueeze(-1)
            return sigmasq
        elif self.nu == 1.5:
            sigmasq = self.kernel_scale.pow(2).unsqueeze(-1).unsqueeze(-1)
            rhosq = self.length_scale.pow(2).unsqueeze(-1).unsqueeze(-1)
            p_infinity = self.mask00 + (3.0 / rhosq) * self.mask11
            return sigmasq * p_infinity
        elif self.nu == 2.5:
            sigmasq = self.kernel_scale.pow(2).unsqueeze(-1).unsqueeze(-1)
            rhosq = self.length_scale.pow(2).unsqueeze(-1).unsqueeze(-1)
            p_infinity = 0.0
            p_infinity = self.mask00 + \
                (five_thirds / rhosq) * (self.mask11 - self.mask02 - self.mask20) + \
                (25.0 / rhosq.pow(2.0)) * self.mask22
            return sigmasq * p_infinity

    @pyro_method
    def process_covariance(self, A):
        """
        Given a transition matrix `A` computed with `transition_matrix` compute the
        the process covariance as described in Eqn. 3.11 in reference [2].

        :returns torch.Tensor: a batched covariance matrix of shape (num_gps, state_dim, state_dim)
        """
        assert A.shape[-2:] == (self.state_dim, self.state_dim)
        p = self.stationary_covariance()
        q = p - torch.matmul(A.transpose(-1, -2), torch.matmul(p, A))
        return q

    @pyro_method
    def transition_matrix_and_covariance(self, dt):
        """
        Get the transition matrix and process covariance corresponding to a time interval `dt`.

        :param float dt: the time interval over which the GP latent space evolves.
        :returns tuple: (`transition_matrix`, `process_covariance`) both 3-dimensional tensors of
            shape (num_gps, state_dim, state_dim)
        """
        trans_matrix = self.transition_matrix(dt)
        process_covar = self.process_covariance(trans_matrix)
        return trans_matrix, process_covar
