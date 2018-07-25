from __future__ import absolute_import, division, print_function

import pytest
import torch

from pyro.contrib.tracking.extended_kalman_filter import EKFState
from pyro.contrib.tracking.dyamic_models import NcpContinuous, NcvContinuous
from tests.common import assert_equal

from matplotlib import pyplot as plt

import measurements as mm
import dynamic_models as dmm


def test_EKFState_with_NcpContinuous():
    d = 3
    ncp = NcpContinuous(dimension=d, sv2=2.0)
    x = np.random.random(d)
    P = np.eye(d)
    t = 0.0
    dt = 2.0
    ekf_state = EKFState(dynamic_model=ncp, mean=x, cov=P, time=t)

    assert ekf_state.dynamic_model.__class__ == NcpContinuous
    assert ekf_state.dimension == d
    assert ekf_state.dimension_pv == 2*d

    assert np.allclose(x, ekf_state.mean)
    assert np.allclose(P, ekf_state.cov)
    assert np.allclose(x, ekf_state.mean_pv[:d])
    assert np.allclose(P, ekf_state.cov_pv[:d, :d])
    assert np.allclose(t, ekf_state.time)

    ekf_state.init(2*x, 2*P, t + 2.0)
    assert np.allclose(2*x, ekf_state.mean)
    assert np.allclose(2*P, ekf_state.cov)
    assert np.allclose(t + 2.0, ekf_state.time)

    ekf_state.init(2*x, 2*P, t)
    ekf_state1 = ekf_state.copy()
    ekf_state1.predict(dt)
    assert ekf_state1.dynamic_model.__class__ == NcpContinuous

    measurement = mm.PositionMeasurement(
        mean=np.random.random(d),
        cov=np.eye(d),
        time=t + dt)
    chi2_stat = ekf_state1.chi2_stat_of_update(measurement)
    assert chi2_stat >= 0.0
    l = ekf_state1.likelihood_of_update(measurement)
    assert l >= 0.0 and l <= 1.0
    old_mean = ekf_state1.mean.copy()
    dz, S = ekf_state1.update(measurement)
    assert dz.shape == (measurement.dimension,)
    assert S.shape == (measurement.dimension, measurement.dimension)
    assert not np.allclose(ekf_state1.mean, old_mean)

    ekf_state2 = ekf_state1.copy()
    assert ekf_state2.dynamic_model.__class__ == NcpContinuous


def test_EKFState_with_NcvContinuous():
    d = 6
    ncv = NcvContinuous(dimension=d, sa2=2.0)
    x = np.random.random(d)
    P = np.eye(d)
    t = 0.0
    dt = 2.0
    ekf_state = EKFState(
        dynamic_model=ncv, mean=x, cov=P, time=t)

    assert ekf_state.dynamic_model.__class__ == NcvContinuous
    assert ekf_state.dimension == d
    assert ekf_state.dimension_pv == d

    assert np.allclose(x, ekf_state.mean)
    assert np.allclose(P, ekf_state.cov)
    assert np.allclose(x, ekf_state.mean_pv)
    assert np.allclose(P, ekf_state.cov_pv)
    assert np.allclose(t, ekf_state.time)

    ekf_state.init(2*x, 2*P, t + 2.0)
    assert np.allclose(2*x, ekf_state.mean)
    assert np.allclose(2*P, ekf_state.cov)
    assert np.allclose(t + 2.0, ekf_state.time)

    ekf_state.init(2*x, 2*P, t)
    ekf_state1 = ekf_state.copy()
    ekf_state1.predict(dt)
    assert ekf_state1.dynamic_model.__class__ == NcvContinuous

    measurement = mm.PositionMeasurement(
        mean=np.random.random(d),
        cov=np.eye(d),
        time=t + dt)
    chi2_stat = ekf_state1.chi2_stat_of_update(measurement)
    assert chi2_stat >= 0.0
    l = ekf_state1.likelihood_of_update(measurement)
    assert l >= 0.0 and l <= 1.0
    old_mean = ekf_state1.mean.copy()
    dz, S = ekf_state1.update(measurement)
    assert dz.shape == (measurement.dimension,)
    assert S.shape == (measurement.dimension, measurement.dimension)
    assert not np.allclose(ekf_state1.mean, old_mean)

    ekf_state2 = ekf_state1.copy()
    assert ekf_state2.dynamic_model.__class__ == NcvContinuous
