from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints

import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.contrib.tracking.extended_kalman_filter import EKFState
from pyro.contrib.tracking.measurements import PositionMeasurement


class EKFDistribution(TorchDistribution):
    r"""
    EKF Distribution

    """
    arg_constraints = {'measurement_cov': constraints.positive_definite,
                       'P0': constraints.positive_definite,
                       'x0': constraints.real_vector}
    has_rsample = True

    def __init__(self, x0, P0, dynamic_model, measurement_cov, dt=1., validate_args=None):
        self.x0 = x0
        self.P0 = P0
        self.dynamic_model = dynamic_model
        self.measurement_cov = measurement_cov
        self.dt = dt
        batch_shape = x0.shape[:-1]
        event_shape = x0.shape[-1:]
        super(EKFDistribution, self).__init__(batch_shape, event_shape,
                                              validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        raise NotImplementedError('TODO: implement forward filter backward sample')

    def log_prob(self, value):
        state = EKFState(self.dynamic_model, self.x0, self.P0, time=0.)
        result = 0.
        zero = value.new_zeros(self.event_shape)
        for measurement_mean in value:
            state = state.predict(self.dt)
            measurement = PositionMeasurement(measurement_mean, self.measurement_cov,
                                              time=state.time)
            state, (dz, S) = state.update(measurement)
            result = result + dist.MultivariateNormal(dz, S).log_prob(zero)
        return result
