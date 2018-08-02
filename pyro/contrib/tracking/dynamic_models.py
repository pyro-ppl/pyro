from abc import ABC, abstractmethod

import torch
import pyro.distributions as dist


class DynamicModel(ABC):
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
    def __call__(self, x, dt, do_normalization=True):
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
        pass

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

        .. warning:: For efficiency, may return a reference to the input.
            Deepcopy as necessary to prevent unexpected changes.

        :param x: native state estimate mean.
        :return: PV state estimate mean.
        '''
        pass

    @abstractmethod
    def cov2pv(self, P):
        '''
        Compute and return PV covariance from native covariance. Useful for
        combining state estimates of different types in IMM (Interacting
        Multiple Model) filtering.

        .. warning:: For efficiency, may return a reference to the input.
            Deepcopy as necessary to prevent unexpected changes.

        :param P: native state estimate covariance.
        :return: PV state estimate covariance.
        '''
        pass

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
        pass

    def sample_process_noise(self, dt=0.):
        '''
        Sample and return a state displacement from the process noise
        distribution over a time interval.

        :param dt: time interval that process noise accumulates over.
        :return: State displacement.
        '''
        Q = self.process_noise_cov(dt)
        dx = dist.MultivariateNormal(torch.zeros(Q.shape[-1]), Q).sample()
        return dx

    @abstractmethod
    def copy(self):
        pass


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
        pass


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
        self._sv2 = sv2
        self._F_cache = torch.eye(dimension)  # State transition matrix cache
        self._Q_cache = {}  # Process noise cov cache

    def __call__(self, x, dt, do_normalization=True):
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
        return x.clone()

    def mean2pv(self, x):
        '''
        Compute and return PV state from native state. Useful for combining
        state estimates of different types in IMM (Interacting Multiple Model)
        filtering.

        :param x: native state estimate mean.
        :return: PV state estimate mean.
        '''
        x_pv = torch.zeros(2*self._dimension)
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
        P_pv = torch.zeros((d, d))
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
        pass


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
        self._sa2 = sa2
        self._F_cache = {}  # State transition matrix cache
        self._Q_cache = {}  # Process noise cov cache

    def __call__(self, x, dt, do_normalization=True):
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

        .. warning:: For efficiency, returns a reference to the input. Deepcopy
            as necessary to prevent unexpected changes.

        :param x: native state estimate mean.
        :return: PV state estimate mean.
        '''
        return x

    def cov2pv(self, P):
        '''
        Compute and return PV covariance from native covariance. Useful for
        combining state estimates of different types in IMM (Interacting
        Multiple Model) filtering.

        .. warning:: For efficiency, returns a reference to the input. Deepcopy
            as necessary to prevent unexpected changes.

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
            F = torch.eye(d)
            F[:d//2, d//2:] = dt*torch.eye(d//2)
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
        pass


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
            q = self._sv2 * dt
            Q = q * dt * torch.eye(self._dimension)
            self._Q_cache[dt] = Q

        return self._Q_cache[dt]

    def copy(self):
        '''
        Deepcopy, except does not copy cached data
        .'''
        return NcpContinuous(self._dimension, self._sv2)


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
            # q: continuous-time process noise intensity with units
            #   length^2/time^3 (m^2/s^3). Choose ``q`` so that changes in
            #   velocity, over a sampling period ``dt``, are roughly
            #   ``sqrt(q*dt)``.
            q = self._sa2 * dt
            d = self._dimension
            dt2 = dt * dt
            dt3 = dt2 * dt
            Q = torch.zeros((d, d))
            Q[:d//2, :d//2] = dt3 * torch.eye(d//2) / 3.0
            Q[:d//2, d//2:] = dt2 * torch.eye(d//2) / 2.0
            Q[d//2:, :d//2] = dt2 * torch.eye(d//2) / 2.0
            Q[d//2:, d//2:] = dt * torch.eye(d//2)
            Q *= q
            self._Q_cache[dt] = Q

        return self._Q_cache[dt]

    def copy(self):
        '''
        Deepcopy, except does not copy cached data.
        '''
        return NcvContinuous(self._dimension, self._sa2)
