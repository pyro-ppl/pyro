# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.contrib.tracking.extended_kalman_filter import EKFState
from pyro.contrib.tracking.measurements import PositionMeasurement


class EKFDistribution(TorchDistribution):
    r"""
    Distribution over EKF states.  See :class:`~pyro.contrib.tracking.extended_kalman_filter.EKFState`.
    Currently only supports `log_prob`.

    :param x0: PV tensor (mean)
    :type x0: torch.Tensor
    :param P0: covariance
    :type P0: torch.Tensor
    :param dynamic_model: :class:`~pyro.contrib.tracking.dynamic_models.DynamicModel` object
    :param measurement_cov: measurement covariance
    :type measurement_cov: torch.Tensor
    :param time_steps: number time step
    :type time_steps: int
    :param dt: time step
    :type dt: torch.Tensor
    """
    arg_constraints = {'measurement_cov': constraints.positive_definite,
                       'P0': constraints.positive_definite,
                       'x0': constraints.real_vector}
    has_rsample = True

    def __init__(self, x0, P0, dynamic_model, measurement_cov, time_steps=1, dt=1., validate_args=None):
        self.x0 = x0
        self.P0 = P0
        self.dynamic_model = dynamic_model
        self.measurement_cov = measurement_cov
        self.dt = dt
        assert not x0.shape[-1] % 2, 'position and velocity vectors must be the same dimension'
        batch_shape = x0.shape[:-1]
        event_shape = (time_steps, x0.shape[-1] // 2)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError('TODO: implement forward filter backward sample')

    def filter_states(self, value):
        """
        Returns the ekf states given measurements

        :param value: measurement means of shape `(time_steps, event_shape)`
        :type value: torch.Tensor
        """
        states = []
        state = EKFState(self.dynamic_model, self.x0, self.P0, time=0.)
        assert value.shape[-1] == self.event_shape[-1]
        for i, measurement_mean in enumerate(value):
            if i:
                state = state.predict(self.dt)
            measurement = PositionMeasurement(measurement_mean, self.measurement_cov,
                                              time=state.time)
            state, (dz, S) = state.update(measurement)
            states.append(state)
        return states

    def log_prob(self, value):
        """
        Returns the joint log probability of the innovations of a tensor of measurements

        :param value: measurement means of shape `(time_steps, event_shape)`
        :type value: torch.Tensor
        """
        state = EKFState(self.dynamic_model, self.x0, self.P0, time=0.)
        result = 0.
        assert value.shape == self.event_shape
        zero = torch.zeros(self.event_shape[-1], dtype=value.dtype, device=value.device)
        for i, measurement_mean in enumerate(value):
            if i:
                state = state.predict(self.dt)
            measurement = PositionMeasurement(measurement_mean, self.measurement_cov,
                                              time=state.time)
            state, (dz, S) = state.update(measurement)
            result = result + dist.MultivariateNormal(dz, S).log_prob(zero)
        return result
