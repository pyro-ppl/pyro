import torch

import pyro.distributions as dist


class EKFState(object):
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
    def __init__(self, dynamic_model):
        self._dynamic_model = dynamic_model
        super(EKFState, self).__init__()

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
    def dimension_pv(self):
        '''
        PV state dimension access.
        '''
        return self._dynamic_model.dimension_pv

    def mean_pv(self, mean):
        '''
        Compute and return cached PV state estimate mean.
        '''
        return self._dynamic_model.mean2pv(mean)

    def cov_pv(self, cov):
        '''
        Compute and return cached PV state estimate covariance.
        '''
        return self._dynamic_model.cov2pv(cov)

    def predict(self, mean, cov, dt=None):
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
        pred_mean = self._dynamic_model(mean, dt)

        F = self._dynamic_model.jacobian(dt)
        Q = self._dynamic_model.process_noise_cov(dt)
        pred_cov = F.mm(cov).mm(F.transpose(-1, -2)) + Q

        return pred_mean, pred_cov

    def innovation(self, mean, cov, measurement):
        '''
        Compute and return the innovation that a measurement would induce if
        it were used for an update, but don't actually perform the update.
        Assumes state and measurement are time-aligned. Useful for computing
        Chi^2 stats and likelihoods.

        :param measurement: measurement
        :return: Innovation mean and covariance of hypothetical update.
        :rtype: tuple(``torch.Tensor``, ``torch.Tensor``)
        '''
        # Compute innovation.
        x_pv = self._dynamic_model.mean2pv(mean)
        H = measurement.jacobian(x_pv)[:, :self.dimension]
        R = measurement.cov
        z = measurement.mean
        z_predicted = measurement(x_pv)
        dz = measurement.geodesic_difference(z, z_predicted)
        S = H.mm(cov).mm(H.transpose(-1, -2)) + R  # innovation cov

        return dz, S

    def log_likelihood_of_update(self, mean, cov, measurement):
        '''
        Compute and return the likelihood of a potential update, but don't
        actually perform the update. Assumes state and measurement are time-
        aligned. Useful for gating and calculating costs in assignment problems
        for data association.

        :param: measurement.
        :return: Likelihood of hypothetical update.
        '''
        dz, S = self.innovation(mean, cov, measurement)
        return dist.MultivariateNormal(S.new_zeros(S.shape[-1]),
                                       S).log_prob(dz)

    def update(self, mean, cov, measurement):
        '''
        Use measurement to update state estimate in-place and return
        innovation. The innovation is useful, e.g., for evaluating filter
        consistency or updating model likelihoods when the ``EKFState`` is part
        of an ``IMMFState``.

        :param: measurement.
        :returns: Innovation mean and covariance.
        '''
        x, P = mean, cov
        x_pv = self._dynamic_model.mean2pv(x)
        H = measurement.jacobian(x_pv)[:, :self.dimension]
        R = measurement.cov
        z = measurement.mean
        z_predicted = measurement(x_pv)
        dz = measurement.geodesic_difference(z, z_predicted)
        S = H.mm(P).mm(H.transpose(-1, -2)) + R  # innovation cov

        K_prefix = cov.mm(H.transpose(-1, -2))
        dx = K_prefix.mm(torch.gesv(dz, S)[0]).squeeze(1)  # K*dz
        x = self._dynamic_model.geodesic_difference(x, -dx)

        I = torch.eye(self._dynamic_model.dimension)  # noqa: E741
        ImKH = I - K_prefix.mm(torch.gesv(H, S)[0])
        # *Joseph form* of covariance update for numerical stability.
        P = ImKH.mm(cov).mm(ImKH.transpose(-1, -2)) \
            + K_prefix.mm(torch.gesv((K_prefix.mm(torch.gesv(R, S)[0])).transpose(-1, -2),
                          S)[0])

        return x, P, dz, S
