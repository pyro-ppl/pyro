# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta, abstractmethod

import torch
from pyro.distributions.util import eye_like


class Measurement(object, metaclass=ABCMeta):
    '''
    Gaussian measurement interface.

    :param mean: mean of measurement distribution.
    :param cov: covariance of measurement distribution.
    :param time: continuous time of measurement. If this is not
          provided, `frame_num` must be.
    :param frame_num: discrete time of measurement. If this is not
          provided, `time` must be.
    '''
    def __init__(self, mean, cov, time=None, frame_num=None):
        self._dimension = len(mean)
        self._mean = mean
        self._cov = cov
        if time is None and frame_num is None:
            raise ValueError('Must provide time or frame_num!')
        self._time = time
        self._frame_num = frame_num

    @property
    def dimension(self):
        '''
        Measurement space dimension access.
        '''
        return self._dimension

    @property
    def mean(self):
        '''
        Measurement mean (``z`` in most Kalman Filtering literature).
        '''
        return self._mean

    @property
    def cov(self):
        '''
        Noise covariance (``R`` in most Kalman Filtering literature).
        '''
        return self._cov

    @property
    def time(self):
        '''
        Continuous time of measurement.
        '''
        return self._time

    @property
    def frame_num(self):
        '''
        Discrete time of measurement.
        '''
        return self._frame_num

    @abstractmethod
    def __call__(self, x, do_normalization=True):
        '''
        Measurement map (h) for predicting a measurement ``z`` from target
        state ``x``.

        :param x: PV state.
        :param do_normalization: whether to normalize output, e.g., mod'ing angles
              into an interval.
        :return Measurement predicted from state ``x``.
        '''
        raise NotImplementedError

    def geodesic_difference(self, z1, z0):
        '''
        Compute and return the geodesic difference between 2 measurements.
        This is a generalization of the Euclidean operation ``z1 - z0``.

        :param z1: measurement.
        :param z0: measurement.
        :return: Geodesic difference between ``z1`` and ``z2``.
        '''
        return z1 - z0  # Default to Euclidean behavior.


class DifferentiableMeasurement(Measurement):
    '''
    Interface for Gaussian measurement for which Jacobians can be efficiently
    calculated, usu. analytically or by automatic differentiation.
    '''
    @abstractmethod
    def jacobian(self, x=None):
        '''
        Compute and return Jacobian (H) of measurement map (h) at target PV
        state ``x`` .

        :param x: PV state. Use default argument ``None`` when the Jacobian is not
              state-dependent.
        :return: Read-only Jacobian (H) of measurement map (h).
        '''
        raise NotImplementedError


class PositionMeasurement(DifferentiableMeasurement):
    '''
    Full-rank Gaussian position measurement in Euclidean space.

    :param mean: mean of measurement distribution.
    :param cov: covariance of measurement distribution.
    :param time: time of measurement.
    '''
    def __init__(self, mean, cov, time=None, frame_num=None):
        super().__init__(mean, cov, time=time, frame_num=frame_num)
        self._jacobian = torch.cat([
            eye_like(mean, self.dimension),
            torch.zeros(self.dimension, self.dimension, dtype=mean.dtype, device=mean.device)], dim=1)

    def __call__(self, x, do_normalization=True):
        '''
        Measurement map (h) for predicting a measurement ``z`` from target
        state ``x``.

        :param x: PV state.
        :param do_normalization: whether to normalize output. Has no effect for
              this subclass.
        :return: Measurement predicted from state ``x``.
        '''
        return x[:self._dimension]

    def jacobian(self, x=None):
        '''
        Compute and return Jacobian (H) of measurement map (h) at target PV
        state ``x`` .

        :param x: PV state. The default argument ``None`` may be used in this
              subclass since the Jacobian is not state-dependent.
        :return: Read-only Jacobian (H) of measurement map (h).
        '''
        return self._jacobian
