from __future__ import absolute_import, division, print_function

import torch

from pyro.contrib.tracking.extended_kalman_filter import EKFState
from pyro.contrib.tracking.dynamic_models import NcpContinuous, NcvContinuous
from pyro.contrib.tracking.measurements import PositionMeasurement
from tests.common import assert_equal, assert_not_equal


def test_EKFState_with_NcpContinuous():
    d = 3
    ncp = NcpContinuous(dimension=d, sv2=2.0)
    mean = torch.rand(d)
    cov = torch.eye(d)
    t = 0.0
    dt = 2.0
    ekf_state = EKFState(dynamic_model=ncp)

    assert ekf_state.dynamic_model.__class__ == NcpContinuous
    assert ekf_state.dimension == d
    assert ekf_state.dimension_pv == 2*d

    assert_equal(mean, ekf_state.mean_pv(mean)[:d], prec=1e-5)
    assert_equal(cov, ekf_state.cov_pv(cov)[:d, :d], prec=1e-5)

    ekf_state.predict(mean, cov, dt=dt)

    measurement = PositionMeasurement(
        mean=torch.rand(d),
        cov=torch.eye(d),
        time=t + dt)
    likelihood = ekf_state.log_likelihood_of_update(mean, cov, measurement)
    assert (likelihood < 0.).all()
    old_mean = mean.clone()
    old_cov = cov.clone()
    mean, cov, dz, S = ekf_state.update(mean, cov, measurement)
    assert dz.shape == (measurement.dimension,)
    assert S.shape == (measurement.dimension, measurement.dimension)
    assert_not_equal(mean, old_mean, prec=1e-5)
    assert_not_equal(cov, old_cov, prec=1e-5)


def test_EKFState_with_NcvContinuous():
    d = 6
    ncv = NcvContinuous(dimension=d, sa2=2.0)
    mean = torch.rand(d)
    cov = torch.eye(d)
    t = 0.0
    dt = 2.0
    ekf_state = EKFState(
        dynamic_model=ncv)

    assert ekf_state.dynamic_model.__class__ == NcvContinuous
    assert ekf_state.dimension == d
    assert ekf_state.dimension_pv == d

    assert_equal(mean, ekf_state.mean_pv(mean), prec=1e-5)
    assert_equal(cov, ekf_state.cov_pv(cov), prec=1e-5)

    ekf_state.predict(mean, cov, dt=dt)

    measurement = PositionMeasurement(
        mean=torch.rand(d),
        cov=torch.eye(d),
        time=t + dt)
    likelihood = ekf_state.log_likelihood_of_update(mean, cov, measurement)
    assert (likelihood < 0.).all()
    old_mean = mean.clone()
    old_cov = cov.clone()
    mean, cov, dz, S = ekf_state.update(mean, cov, measurement)
    assert dz.shape == (measurement.dimension,)
    assert S.shape == (measurement.dimension, measurement.dimension)
    assert_not_equal(mean, old_mean, prec=1e-5)
    assert_not_equal(cov, old_cov, prec=1e-5)
