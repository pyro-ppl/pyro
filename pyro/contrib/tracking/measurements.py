from abc import ABC, abstractmethod

import torch


class Measurement(ABC):
    '''
    Gaussian measurement interface.

    :param mean: mean of measurement distribution.
    :param cov: covariance of measurement distribution.
    :param time: time of measurement.
    '''
    def __init__(self, mean, cov, time):
        self._dimension = len(mean)
        self._mean = mean.clone()
        self._cov = cov.clone()
        self._time = time

    @property
    def dimension(self):
        '''
        Measurement space dimension access.
        '''
        return self._dimension

    @property
    def mean(self):
        '''
        Measurement mean (`z` in most Kalman Filtering literature).
        '''
        return self._mean

    @property
    def cov(self):
        '''
        Noise covariance (`R` in most Kalman Filtering literature).
        '''
        return self._cov

    @property
    def time(self):
        '''
        Time of measurement.
        '''
        return self._time

    @abstractmethod
    def __call__(self, x, do_normalization=True):
        '''
        Measurement map (h) for predicting a measurement `z` from target
        state `x`.

        :param x: PV state.
        :param do_normalization: whether to normalize output, e.g., mod'ing angles
              into an interval.
        :return Measurement predicted from state `x`.
        '''
        pass

    def geodesic_difference(self, z1, z0):
        '''
        Compute and return the geodesic difference between 2 measurements.
        This is a generalization of the Euclidean operation `z1 - z0`.

        :param z1: measurement.
        :param z0: measurement.
        :return: Geodesic difference between `z1` and `z2`.
        '''
        return z1 - z0  # Default to Euclidean behavior.

    @abstractmethod
    def copy(self):
        pass


class DifferentiableMeasurement(Measurement):
    '''
    Interface for Gaussian measurement for which Jacobians can be efficiently
    calculated, usu. analytically or by automatic differentiation.
    '''
    @abstractmethod
    def jacobian(self, x=None):
        '''
        Compute and return Jacobian (H) of measurement map (h) at target PV
        state `x` .

        :param x: PV state. Use default argument `None` when the Jacobian is not
              state-dependent.
        :return: Read-only Jacobian (H) of measurement map (h).
        '''
        pass


class PositionMeasurement(DifferentiableMeasurement):
    '''
    Full-rank Gaussian position measurement in Euclidean space.

    :param mean: mean of measurement distribution.
    :param cov: covariance of measurement distribution.
    :param time: time of measurement.
    '''
    def __init__(self, mean, cov, time):
        super().__init__(mean, cov, time)
        self._jacobian = torch.cat([
            torch.eye(self.dimension),
            torch.zeros((self.dimension, self.dimension))], dim=1)

    def __call__(self, x, do_normalization=True):
        '''
        Measurement map (h) for predicting a measurement `z` from target
        state `x`.

        :param x: PV state.
        :param do_normalization: whether to normalize output. Has no effect for
              this subclass.
        :return: Measurement predicted from state `x`.
        '''
        return x[:self._dimension].clone()

    def jacobian(self, x=None):
        '''
        Compute and return Jacobian (H) of measurement map (h) at target PV
        state `x` .

        :param x: PV state. The default argument `None` may be used in this
              subclass since the Jacobian is not state-dependent.
        :return: Read-only Jacobian (H) of measurement map (h).
        '''
        return self._jacobian

    def copy(self):
        return PositionMeasurement(self._mean, self._cov, self._time)
