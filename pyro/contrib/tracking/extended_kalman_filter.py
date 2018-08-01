import torch
import pyro.distributions as dist


class EKFState:
    '''
    State-Centric EKF (Extended Kalman Filter) for use with either an NCP
    (Nearly-Constant Position) or NCV (Nearly-Constant Velocity) target dynamic
    model. Stores a target dynamic model, state estimate, and state time.
    Incoming `Measurement`s provide sensor information for updates.

    .. warning:: For efficiency, the dynamic model is only shallow-copied. Make
    a deep copy outside as necessary to protect against unexpected
    changes.

    :param dynamic_model: target dynamic model.
    :param mean: mean of target state estimate.
    :param cov: covariance of target state estimate.
    :param time: time of state estimate.
    '''
    def __init__(self, dynamic_model, mean, cov, time):
        self._dynamic_model = dynamic_model
        if mean is None:
            self._mean = None
        else:
            self._mean = mean.clone()
        if cov is None:
            self._cov = None
        else:
            self._cov = cov.clone()
        self._time = time

        self._mean_pv_cache = None
        self._cov_pv_cache = None

    def _clear_cached(self):
        '''
        Call this whenever actions are taken which invalidate cached data.
        '''
        self._mean_pv_cache = None
        self._cov_pv_cache = None

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

    @property
    def mean_pv(self):
        '''
        Compute and return cached PV state estimate mean.
        '''
        if self._mean_pv_cache is None:
            self._mean_pv_cache = \
                self._dynamic_model.mean2pv(self._mean)

        return self._mean_pv_cache

    @property
    def cov_pv(self):
        '''
        Compute and return cached PV state estimate covariance.
        '''
        if self._cov_pv_cache is None:
            self._cov_pv_cache = \
                self._dynamic_model.cov2pv(self._cov)

        return self._cov_pv_cache

    @property
    def time(self):
        '''
        State time access.
        '''
        return self._time

    def init(self, mean, cov, time):
        '''
        Re-initialize target state.

        :param mean: target state mean.
        :param cov: target state covariance.
        :param time: state time. None => keep existing time.
        '''
        self._mean = mean.clone()
        self._cov = cov.clone()
        if time is not None:
            self._time = time

        self._clear_cached()

    def predict(self, dt=None, destination_time=None):
        '''
        Use dynamic model to predict (aka propagate aka integrate) state
        estimate in-place.

        :param dt: time to integrate over. The state time will be automatically
                   incremented this amount unless you provide `destination_time`.
                   Using `destination_time` may be preferable for prevention of
                   roundoff error accumulation.
        :param destination_time: time to increment to.
        '''
        if dt is not None and destination_time is not None:
            assert ((destination_time - self._time + dt) < 1e-3).all()
        elif dt is None:
            assert destination_time is not None
            dt = destination_time - self._time
        elif destination_time is None:
            assert dt is not None
            destination_time = self._time + dt

        self._mean = self._dynamic_model(self._mean, dt)

        F = self._dynamic_model.jacobian(dt)
        Q = self._dynamic_model.process_noise_cov(dt)
        self._cov = F.mm(self._cov).mm(F.t()) + Q

        self._time = destination_time

        self._clear_cached()

    def innovation(self, measurement):
        '''
        Compute and return the innovation that a measurement would induce if
        it were used for an update, but don't actually perform the update.
        Assumes state and measurement are time-aligned. Useful for computing
        Chi^2 stats and likelihoods.

        :param measurement: measurement
        :return: Innovation mean and covariance of hypothetical update.
        :rtype: tuple(torch.Tensor, torch.Tensor)
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
        S = H.mm(self._cov).mm(H.t()) + R  # innovation cov

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
        return torch.exp(dist.MultivariateNormal(torch.zeros_like(S), S)
                         .log_prob(dz))

    def update(self, measurement):
        '''
        Use measurement to update state estimate in-place and return
        innovation. The innovation is useful, e.g., for evaluating filter
        consistency or updating model likelihoods when the `EKFState` is part
        of an `IMMFState`.

        :param: measurement.
        :returns: Innovation mean and covariance.
        '''
        assert self._time == measurement.time, \
            'State time and measurement time must be aligned!'

        x = self._mean
        x_pv = self._dynamic_model.mean2pv(x)
        P = self.cov
        H = measurement.jacobian(x_pv)[:, :self.dimension]
        R = measurement.cov
        z = measurement.mean
        z_predicted = measurement(x_pv)
        dz = measurement.geodesic_difference(z, z_predicted)
        S = H.mm(P).mm(H.t()) + R  # innovation cov

        K_prefix = self._cov.mm(H.t())
        dx = K_prefix.mm(torch.gesv(dz, S)[0])  # K*dz
        x = self._dynamic_model.geodesic_difference(x, -dx)

        I = torch.eye(self._dynamic_model.dimension)  # noqa: E741
        ImKH = I - K_prefix.mm(torch.gesv(H, S)[0])
        # *Joseph form* of covariance update for numerical stability.
        P = ImKH.mm(self.cov).mm(ImKH.t()) \
            + K_prefix.mm(torch.gesv((K_prefix.mm(torch.gesv(R, S)[0])).t(),
                          S)[0])

        self._mean = x
        self._cov = P

        self._clear_cached()

        return dz, S

    def copy(self):
        '''
        Deepcopy everything, except dynamic model is only shallow-copied.
        '''
        return EKFState(
            dynamic_model=self._dynamic_model,
            mean=self._mean, cov=self._cov, time=self._time)
