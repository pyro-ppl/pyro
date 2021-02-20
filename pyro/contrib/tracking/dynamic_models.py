# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta, abstractmethod

import torch
from torch import nn
from torch.nn import Parameter
import pyro.distributions as dist
from pyro.distributions.util import eye_like


class DynamicModel(nn.Module, metaclass=ABCMeta):
    '''
    Dynamic model interface.

    :param dimension: native state dimension.
    :param dimension_pv: PV state dimension.
    :param num_process_noise_parameters: process noise parameter space dimension.
          This for UKF applications. Can be left as ``None`` for EKF and most
          other filters.
    '''
    def __init__(self, dimension, dimension_pv, num_process_noise_parameters=None):
        self._dimension = dimension
        self._dimension_pv = dimension_pv
        self._num_process_noise_parameters = num_process_noise_parameters
        super().__init__()

    @property
    def dimension(self):
        '''
        Native state dimension access.
        '''
        return self._dimension

    @property
    def dimension_pv(self):
        '''
        PV state dimension access.
        '''
        return self._dimension_pv

    @property
    def num_process_noise_parameters(self):
        '''
        Process noise parameters space dimension access.
        '''
        return self._num_process_noise_parameters

    @abstractmethod
    def forward(self, x, dt, do_normalization=True):
        '''
        Integrate native state ``x`` over time interval ``dt``.

        :param x: current native state. If the DynamicModel is non-differentiable,
              be sure to handle the case of ``x`` being augmented with process
              noise parameters.
        :param dt: time interval to integrate over.
        :param do_normalization: whether to perform normalization on output, e.g.,
              mod'ing angles into an interval.
        :return: Native state x integrated dt into the future.
        '''
        raise NotImplementedError

    def geodesic_difference(self, x1, x0):
        '''
        Compute and return the geodesic difference between 2 native states.
        This is a generalization of the Euclidean operation ``x1 - x0``.

        :param x1: native state.
        :param x0: native state.
        :return: Geodesic difference between native states ``x1`` and ``x2``.
        '''
        return x1 - x0  # Default to Euclidean behavior.

    @abstractmethod
    def mean2pv(self, x):
        '''
        Compute and return PV state from native state. Useful for combining
        state estimates of different types in IMM (Interacting Multiple Model)
        filtering.

        :param x: native state estimate mean.
        :return: PV state estimate mean.
        '''
        raise NotImplementedError

    @abstractmethod
    def cov2pv(self, P):
        '''
        Compute and return PV covariance from native covariance. Useful for
        combining state estimates of different types in IMM (Interacting
        Multiple Model) filtering.

        :param P: native state estimate covariance.
        :return: PV state estimate covariance.
        '''
        raise NotImplementedError

    @abstractmethod
    def process_noise_cov(self, dt=0.):
        '''
        Compute and return process noise covariance (Q).

        :param dt: time interval to integrate over.
        :return: Read-only covariance (Q). For a DifferentiableDynamicModel, this is
            the covariance of the native state ``x`` resulting from stochastic
            integration (for use with EKF). Otherwise, it is the covariance
            directly of the process noise parameters (for use with UKF).
        '''
        raise NotImplementedError

    def process_noise_dist(self, dt=0.):
        '''
        Return a distribution object of state displacement from the process noise
        distribution over a time interval.

        :param dt: time interval that process noise accumulates over.
        :return: :class:`~pyro.distributions.torch.MultivariateNormal`.
        '''
        Q = self.process_noise_cov(dt)
        return dist.MultivariateNormal(torch.zeros(Q.shape[-1], dtype=Q.dtype, device=Q.device), Q)


class DifferentiableDynamicModel(DynamicModel):
    '''
    DynamicModel for which state transition Jacobians can be efficiently
    calculated, usu. analytically or by automatic differentiation.
    '''
    @abstractmethod
    def jacobian(self, dt):
        '''
        Compute and return native state transition Jacobian (F) over time
        interval ``dt``.

        :param  dt: time interval to integrate over.
        :return: Read-only Jacobian (F) of integration map (f).
        '''
        raise NotImplementedError


class Ncp(DifferentiableDynamicModel):
    '''
    NCP (Nearly-Constant Position) dynamic model. May be subclassed, e.g., with
    CWNV (Continuous White Noise Velocity) or DWNV (Discrete White Noise
    Velocity).

    :param dimension: native state dimension.
    :param sv2: variance of velocity. Usually chosen so that the standard
          deviation is roughly half of the max velocity one would ever expect
          to observe.
    '''
    def __init__(self, dimension, sv2):
        dimension_pv = 2 * dimension
        super().__init__(dimension, dimension_pv, num_process_noise_parameters=1)
        if not isinstance(sv2, torch.Tensor):
            sv2 = torch.tensor(sv2)
        self.sv2 = Parameter(sv2)
        self._F_cache = eye_like(sv2, dimension)  # State transition matrix cache
        self._Q_cache = {}  # Process noise cov cache

    def forward(self, x, dt, do_normalization=True):
        '''
        Integrate native state ``x`` over time interval ``dt``.

        :param x: current native state. If the DynamicModel is non-differentiable,
              be sure to handle the case of ``x`` being augmented with process
              noise parameters.
        :param dt: time interval to integrate over.
            do_normalization: whether to perform normalization on output, e.g.,
            mod'ing angles into an interval. Has no effect for this subclass.
        :return: Native state x integrated dt into the future.
        '''
        return x

    def mean2pv(self, x):
        '''
        Compute and return PV state from native state. Useful for combining
        state estimates of different types in IMM (Interacting Multiple Model)
        filtering.

        :param x: native state estimate mean.
        :return: PV state estimate mean.
        '''
        with torch.no_grad():
            x_pv = torch.zeros(2 * self._dimension, dtype=x.dtype, device=x.device)
            x_pv[:self._dimension] = x
        return x_pv

    def cov2pv(self, P):
        '''
        Compute and return PV covariance from native covariance. Useful for
        combining state estimates of different types in IMM (Interacting
        Multiple Model) filtering.

        :param P: native state estimate covariance.
        :return: PV state estimate covariance.
        '''
        d = 2*self._dimension
        with torch.no_grad():
            P_pv = torch.zeros(d, d, dtype=P.dtype, device=P.device)
            P_pv[:self._dimension, :self._dimension] = P
        return P_pv

    def jacobian(self, dt):
        '''
        Compute and return cached native state transition Jacobian (F) over
        time interval ``dt``.

        :param dt: time interval to integrate over.
        :return: Read-only Jacobian (F) of integration map (f).
        '''
        return self._F_cache

    @abstractmethod
    def process_noise_cov(self, dt=0.):
        '''
        Compute and return cached process noise covariance (Q).

        :param dt: time interval to integrate over.
        :return: Read-only covariance (Q) of the native state ``x`` resulting from
            stochastic integration (for use with EKF).
        '''
        raise NotImplementedError


class Ncv(DifferentiableDynamicModel):
    '''
    NCV (Nearly-Constant Velocity) dynamic model. May be subclassed, e.g., with
    CWNA (Continuous White Noise Acceleration) or DWNA (Discrete White Noise
    Acceleration).

    :param dimension: native state dimension.
    :param sa2: variance of acceleration. Usually chosen so that the standard
          deviation is roughly half of the max acceleration one would ever
          expect to observe.
    '''
    def __init__(self, dimension, sa2):
        dimension_pv = dimension
        super().__init__(dimension, dimension_pv, num_process_noise_parameters=1)
        if not isinstance(sa2, torch.Tensor):
            sa2 = torch.tensor(sa2)
        self.sa2 = Parameter(sa2)
        self._F_cache = {}  # State transition matrix cache
        self._Q_cache = {}  # Process noise cov cache

    def forward(self, x, dt, do_normalization=True):
        '''
        Integrate native state ``x`` over time interval ``dt``.

        :param x: current native state. If the DynamicModel is non-differentiable,
              be sure to handle the case of ``x`` being augmented with process
              noise parameters.
        :param dt: time interval to integrate over.
        :param do_normalization: whether to perform normalization on output, e.g.,
              mod'ing angles into an interval. Has no effect for this subclass.

        :return: Native state x integrated dt into the future.
        '''
        F = self.jacobian(dt)
        return F.mm(x.unsqueeze(1)).squeeze(1)

    def mean2pv(self, x):
        '''
        Compute and return PV state from native state. Useful for combining
        state estimates of different types in IMM (Interacting Multiple Model)
        filtering.

        :param x: native state estimate mean.
        :return: PV state estimate mean.
        '''
        return x

    def cov2pv(self, P):
        '''
        Compute and return PV covariance from native covariance. Useful for
        combining state estimates of different types in IMM (Interacting
        Multiple Model) filtering.

        :param P: native state estimate covariance.
        :return: PV state estimate covariance.
        '''
        return P

    def jacobian(self, dt):
        '''
        Compute and return cached native state transition Jacobian (F) over
        time interval ``dt``.

        :param dt: time interval to integrate over.
        :return: Read-only Jacobian (F) of integration map (f).
        '''
        if dt not in self._F_cache:
            d = self._dimension
            with torch.no_grad():
                F = eye_like(self.sa2, d)
                F[:d//2, d//2:] = dt * eye_like(self.sa2, d//2)
            self._F_cache[dt] = F

        return self._F_cache[dt]

    @abstractmethod
    def process_noise_cov(self, dt=0.):
        '''
        Compute and return cached process noise covariance (Q).

        :param dt: time interval to integrate over.
        :return: Read-only covariance (Q) of the native state ``x`` resulting from
            stochastic integration (for use with EKF).
        '''
        raise NotImplementedError


class NcpContinuous(Ncp):
    '''
    NCP (Nearly-Constant Position) dynamic model with CWNV (Continuous White
    Noise Velocity).

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.269.

    :param dimension: native state dimension.
    :param sv2: variance of velocity. Usually chosen so that the standard
          deviation is roughly half of the max velocity one would ever expect
          to observe.
    '''
    def process_noise_cov(self, dt=0.):
        '''
        Compute and return cached process noise covariance (Q).

        :param dt: time interval to integrate over.
        :return: Read-only covariance (Q) of the native state ``x`` resulting from
            stochastic integration (for use with EKF).
        '''
        if dt not in self._Q_cache:
            # q: continuous-time process noise intensity with units
            #   length^2/time (m^2/s). Choose ``q`` so that changes in position,
            #   over a sampling period ``dt``, are roughly ``sqrt(q*dt)``.
            q = self.sv2 * dt
            Q = q * dt * eye_like(self.sv2, self._dimension)
            self._Q_cache[dt] = Q

        return self._Q_cache[dt]


class NcvContinuous(Ncv):
    '''
    NCV (Nearly-Constant Velocity) dynamic model with CWNA (Continuous White
    Noise Acceleration).

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.269.

    :param dimension: native state dimension.
    :param sa2: variance of acceleration. Usually chosen so that the standard
          deviation is roughly half of the max acceleration one would ever
          expect to observe.
    '''
    def process_noise_cov(self, dt=0.):
        '''
        Compute and return cached process noise covariance (Q).

        :param dt: time interval to integrate over.

        :return: Read-only covariance (Q) of the native state ``x`` resulting from
            stochastic integration (for use with EKF).
        '''
        if dt not in self._Q_cache:

            with torch.no_grad():
                d = self._dimension
                dt2 = dt * dt
                dt3 = dt2 * dt
                Q = torch.zeros(d, d, dtype=self.sa2.dtype, device=self.sa2.device)
                eye = eye_like(self.sa2, d//2)
                Q[:d//2, :d//2] = dt3 * eye / 3.0
                Q[:d//2, d//2:] = dt2 * eye / 2.0
                Q[d//2:, :d//2] = dt2 * eye / 2.0
                Q[d//2:, d//2:] = dt * eye
            # sa2 * dt is an intensity factor that changes in velocity
            # over a sampling period ``dt``, ideally should be ~``sqrt(q*dt)``.
            Q = Q * (self.sa2 * dt)
            self._Q_cache[dt] = Q

        return self._Q_cache[dt]


class NcpDiscrete(Ncp):
    '''
    NCP (Nearly-Constant Position) dynamic model with DWNV (Discrete White
    Noise Velocity).

    :param dimension: native state dimension.
    :param sv2: variance of velocity. Usually chosen so that the standard
          deviation is roughly half of the max velocity one would ever expect
          to observe.

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.273.
    '''
    def process_noise_cov(self, dt=0.):
        '''
        Compute and return cached process noise covariance (Q).

        :param dt: time interval to integrate over.
        :return: Read-only covariance (Q) of the native state `x` resulting from
            stochastic integration (for use with EKF).
        '''
        if dt not in self._Q_cache:
            Q = self.sv2 * dt * dt * eye_like(self.sv2, self._dimension)
            self._Q_cache[dt] = Q

        return self._Q_cache[dt]


class NcvDiscrete(Ncv):
    '''
    NCV (Nearly-Constant Velocity) dynamic model with DWNA (Discrete White
    Noise Acceleration).

    :param dimension: native state dimension.
    :param sa2: variance of acceleration. Usually chosen so that the standard
          deviation is roughly half of the max acceleration one would ever
          expect to observe.

    References:
        "Estimation with Applications to Tracking and Navigation" by Y. Bar-
        Shalom et al, 2001, p.273.
    '''
    def process_noise_cov(self, dt=0.):
        '''
        Compute and return cached process noise covariance (Q).

        :param dt: time interval to integrate over.
        :return: Read-only covariance (Q) of the native state `x` resulting from
            stochastic integration (for use with EKF). (Note that this Q, modulo
            numerical error, has rank `dimension/2`. So, it is only positive
            semi-definite.)
        '''
        if dt not in self._Q_cache:
            with torch.no_grad():
                d = self._dimension
                dt2 = dt*dt
                dt3 = dt2*dt
                dt4 = dt2*dt2
                Q = torch.zeros(d, d, dtype=self.sa2.dtype, device=self.sa2.device)
                Q[:d//2, :d//2] = 0.25 * dt4 * eye_like(self.sa2, d//2)
                Q[:d//2, d//2:] = 0.5 * dt3 * eye_like(self.sa2, d//2)
                Q[d//2:, :d//2] = 0.5 * dt3 * eye_like(self.sa2, d//2)
                Q[d//2:, d//2:] = dt2 * eye_like(self.sa2, d//2)
            Q = Q * self.sa2
            self._Q_cache[dt] = Q

        return self._Q_cache[dt]
