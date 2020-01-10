# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from pyro.contrib.tracking.measurements import PositionMeasurement


def test_PositionMeasurement():
    dimension = 3
    time = 0.232
    frame_num = 5
    measurement = PositionMeasurement(
        mean=torch.rand(dimension),
        cov=torch.eye(dimension), time=time, frame_num=frame_num)
    assert measurement.dimension == dimension
    x = torch.rand(2*dimension)
    assert measurement(x).shape == (dimension,)
    assert measurement.mean.shape == (dimension,)
    assert measurement.cov.shape == (dimension, dimension)
    assert measurement.time == time
    assert measurement.frame_num == frame_num
    assert measurement.geodesic_difference(
        torch.rand(dimension), torch.rand(dimension)).shape \
        == (dimension,)
    assert measurement.jacobian().shape == (dimension, 2*dimension)
