# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.contrib.tracking.extended_kalman_filter import EKFState
from pyro.contrib.tracking.dynamic_models import NcpContinuous, NcvContinuous
from pyro.contrib.tracking.measurements import PositionMeasurement

from tests.common import assert_equal, assert_not_equal


def test_EKFState_with_NcpContinuous():
    d = 3
    ncp = NcpContinuous(dimension=d, sv2=2.0)
    x = torch.rand(d)
    P = torch.eye(d)
    t = 0.0
    dt = 2.0
    ekf_state = EKFState(dynamic_model=ncp, mean=x, cov=P, time=t)

    assert ekf_state.dynamic_model.__class__ == NcpContinuous
    assert ekf_state.dimension == d
    assert ekf_state.dimension_pv == 2*d

    assert_equal(x, ekf_state.mean, prec=1e-5)
    assert_equal(P, ekf_state.cov, prec=1e-5)
    assert_equal(x, ekf_state.mean_pv[:d], prec=1e-5)
    assert_equal(P, ekf_state.cov_pv[:d, :d], prec=1e-5)
    assert_equal(t, ekf_state.time, prec=1e-5)

    ekf_state1 = EKFState(ncp, 2*x, 2*P, t)
    ekf_state2 = ekf_state1.predict(dt)
    assert ekf_state2.dynamic_model.__class__ == NcpContinuous

    measurement = PositionMeasurement(
        mean=torch.rand(d),
        cov=torch.eye(d),
        time=t + dt)
    log_likelihood = ekf_state2.log_likelihood_of_update(measurement)
    assert (log_likelihood < 0.).all()
    ekf_state3, (dz, S) = ekf_state2.update(measurement)
    assert dz.shape == (measurement.dimension,)
    assert S.shape == (measurement.dimension, measurement.dimension)
    assert_not_equal(ekf_state3.mean, ekf_state2.mean, prec=1e-5)


def test_EKFState_with_NcvContinuous():
    d = 6
    ncv = NcvContinuous(dimension=d, sa2=2.0)
    x = torch.rand(d)
    P = torch.eye(d)
    t = 0.0
    dt = 2.0
    ekf_state = EKFState(
        dynamic_model=ncv, mean=x, cov=P, time=t)

    assert ekf_state.dynamic_model.__class__ == NcvContinuous
    assert ekf_state.dimension == d
    assert ekf_state.dimension_pv == d

    assert_equal(x, ekf_state.mean, prec=1e-5)
    assert_equal(P, ekf_state.cov, prec=1e-5)
    assert_equal(x, ekf_state.mean_pv, prec=1e-5)
    assert_equal(P, ekf_state.cov_pv, prec=1e-5)
    assert_equal(t, ekf_state.time, prec=1e-5)

    ekf_state1 = EKFState(ncv, 2*x, 2*P, t)
    ekf_state2 = ekf_state1.predict(dt)
    assert ekf_state2.dynamic_model.__class__ == NcvContinuous

    measurement = PositionMeasurement(
        mean=torch.rand(d),
        cov=torch.eye(d),
        time=t + dt)
    log_likelihood = ekf_state2.log_likelihood_of_update(measurement)
    assert (log_likelihood < 0.).all()
    ekf_state3, (dz, S) = ekf_state2.update(measurement)
    assert dz.shape == (measurement.dimension,)
    assert S.shape == (measurement.dimension, measurement.dimension)
    assert_not_equal(ekf_state3.mean, ekf_state2.mean, prec=1e-5)
