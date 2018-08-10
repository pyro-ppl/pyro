import torch
from torch.distributions.utils import lazy_property

import pyro.distributions as dist


class EKFState:
    '''
    State-Centric EKF (Extended Kalman Filter) for use with either an NCP
    (Nearly-Constant Position) or NCV (Nearly-Constant Velocity) target dynamic
    model. Stores a target dynamic model, state estimate, and state time.
    Incoming ``Measurement`` provide sensor information for updates.

    .. warning:: For efficiency, the dynamic model is only shallow-copied. Make
        a deep copy outside as necessary to protect against unexpected changes.

    :param dynamic_model: target dynamic model.
    :param mean: mean of target state estimate.
    :param cov: covariance of target state estimate.
    :param time: time of state estimate.
    '''
    def __init__(self, dynamic_model, mean, cov, time=None, frame_num=None):
        self._dynamic_model = dynamic_model
        self._mean = mean
        self._cov = cov
        if time is None and frame_num is None:
            raise ValueError('Must provide time or frame_num!')
        self._time = time
        self._frame_num = frame_num

    def _clear_cached(self):
        '''
        Call this whenever actions are taken which invalidate cached data.
        '''
        # XXX: remove in place mutations and this function
        self.__dict__.pop('mean_pv', None)
        self.__dict__.pop('cov_pv', None)

    @property
    def dynamic_model(self):
        '''
        Dynamic model access.
        '''
        return self._dynamic_model

    @property
    def dimension(self):
        '''
        Native state dimension access.
        '''
        return self._dynamic_model.dimension

    @property
    def mean(self):
        '''
        Native state estimate mean access.
        '''
        return self._mean

    @property
    def cov(self):
        '''
        Native state estimate covariance access.
        '''
        return self._cov

    @property
    def dimension_pv(self):
        '''
        PV state dimension access.
        '''
        return self._dynamic_model.dimension_pv

    @lazy_property
    def mean_pv(self):
        '''
        Compute and return cached PV state estimate mean.
        '''
        return self._dynamic_model.mean2pv(self._mean)
        if self._mean_pv_cache is None:
            self._mean_pv_cache = \
                self._dynamic_model.mean2pv(self._mean)

        return self._mean_pv_cache

    @lazy_property
    def cov_pv(self):
        '''
        Compute and return cached PV state estimate covariance.
        '''
        return self._dynamic_model.cov2pv(self._cov)
        if self._cov_pv_cache is None:
            self._cov_pv_cache = \
                self._dynamic_model.cov2pv(self._cov)

        return self._cov_pv_cache

    @property
    def time(self):
        '''
        Continuous State time access.
        '''
        return self._time

    @property
    def frame_num(self):
        '''
        Discrete State time access.
        '''
        return self._frame_num

    def init(self, mean, cov, time=None, frame_num=None):
        '''
        Re-initialize target state.

        :param mean: target state mean.
        :param cov: target state covariance.
        :param time: continuous state time. None => keep existing time.
        :param time: discrete state time. None => keep existing time.
        '''
        self._mean = mean
        self._cov = cov
        if time is not None:
            self._time = time
        if frame_num is not None:
            self._frame_num = frame_num

        self._clear_cached()

    def predict(self, dt=None, destination_time=None, destination_frame_num=None):
        '''
        Use dynamic model to predict (aka propagate aka integrate) state
        estimate in-place.

        :param dt: time to integrate over. The state time will be automatically
                   incremented this amount unless you provide ``destination_time``.
                   Using ``destination_time`` may be preferable for prevention of
                   roundoff error accumulation.
        :param destination_time: optional value to set continuous state time to
            after integration. If this is not provided, then
            `destination_frame_num` must be.
        :param destination_frame_num: optional value to set discrete state time to
            after integration. If this is not provided, then
            `destination_frame_num` must be.
        '''
        self._mean = self._dynamic_model(self._mean, dt)

        F = self._dynamic_model.jacobian(dt)
        Q = self._dynamic_model.process_noise_cov(dt)
        self._cov = F.mm(self._cov).mm(F.transpose(-1, -2)) + Q

        if destination_time is None and destination_frame_num is None:
            raise ValueError('destination_time or destination_frame_num must be specified!')
        self._time = destination_time
        self._frame_num = destination_frame_num

        self._clear_cached()

    def innovation(self, measurement):
        '''
        Compute and return the innovation that a measurement would induce if
        it were used for an update, but don't actually perform the update.
        Assumes state and measurement are time-aligned. Useful for computing
        Chi^2 stats and likelihoods.

        :param measurement: measurement
        :return: Innovation mean and covariance of hypothetical update.
        :rtype: tuple(``torch.Tensor``, ``torch.Tensor``)
        '''
        assert self._time == measurement.time, \
            'State time and measurement time must be aligned!'

        # Compute innovation.
        x_pv = self._dynamic_model.mean2pv(self._mean)
        H = measurement.jacobian(x_pv)[:, :self.dimension]
        R = measurement.cov
        z = measurement.mean
        z_predicted = measurement(x_pv)
        dz = measurement.geodesic_difference(z, z_predicted)
        S = H.mm(self._cov).mm(H.transpose(-1, -2)) + R  # innovation cov

        return dz, S

    def likelihood_of_update(self, measurement):
        '''
        Compute and return the likelihood of a potential update, but don't
        actually perform the update. Assumes state and measurement are time-
        aligned. Useful for gating and calculating costs in assignment problems
        for data association.

        :param: measurement.
        :return: Likelihood of hypothetical update.
        '''
        dz, S = self.innovation(measurement)
        return torch.exp(dist.MultivariateNormal(S.new_zeros(S.shape[-1]), S)
                         .log_prob(dz))

    def update(self, measurement):
        '''
        Use measurement to update state estimate in-place and return
        innovation. The innovation is useful, e.g., for evaluating filter
        consistency or updating model likelihoods when the ``EKFState`` is part
        of an ``IMMFState``.

        :param: measurement.
        :returns: Innovation mean and covariance.
        '''
        if self._time is not None:
            assert self._time == measurement.time, \
                'State time and measurement time must be aligned!'
        if self._frame_num is not None:
            assert self._frame_num == measurement.frame_num, \
                'State time and measurement time must be aligned!'

        x = self._mean
        x_pv = self._dynamic_model.mean2pv(x)
        P = self.cov
        H = measurement.jacobian(x_pv)[:, :self.dimension]
        R = measurement.cov
        z = measurement.mean
        z_predicted = measurement(x_pv)
        dz = measurement.geodesic_difference(z, z_predicted)
        S = H.mm(P).mm(H.transpose(-1, -2)) + R  # innovation cov

        K_prefix = self._cov.mm(H.transpose(-1, -2))
        dx = K_prefix.mm(torch.gesv(dz, S)[0]).squeeze(1)  # K*dz
        x = self._dynamic_model.geodesic_difference(x, -dx)

        I = torch.eye(self._dynamic_model.dimension)  # noqa: E741
        ImKH = I - K_prefix.mm(torch.gesv(H, S)[0])
        # *Joseph form* of covariance update for numerical stability.
        P = ImKH.mm(self.cov).mm(ImKH.transpose(-1, -2)) \
            + K_prefix.mm(torch.gesv((K_prefix.mm(torch.gesv(R, S)[0])).transpose(-1, -2),
                          S)[0])

        self._mean = x
        self._cov = P

        self._clear_cached()

        return dz, S

    def copy(self, time=None, frame_num=None):
        '''
        Deepcopy everything, except dynamic model is only shallow-copied.

        Optionally `time` and/or `frame_num` can be reset. This is useful,
        e.g., if you want to cache an intial filter state that can be copied
        into newly initialized tracks at different times.
        '''
        if time is not None and frame_num is None:
            return EKFState(
                dynamic_model=self._dynamic_model,
                mean=self._mean, cov=self._cov,
                time=time, frame_num=None)
        if time is None and frame_num is not None:
            return EKFState(
                dynamic_model=self._dynamic_model,
                mean=self._mean, cov=self._cov,
                time=None, frame_num=frame_num)
        return EKFState(
            dynamic_model=self._dynamic_model,
            mean=self._mean, cov=self._cov, time=self._time)
