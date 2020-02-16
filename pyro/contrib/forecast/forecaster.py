# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal, init_to_sample
from pyro.nn.module import PyroModule
from pyro.optim import ClippedAdam

from .prefix import PrefixReplayMessenger, prefix_condition

logger = logging.getLogger(__name__)


class _ForecastingModelMeta(type(PyroModule), ABCMeta):
    pass


class ForecastingModel(PyroModule, metaclass=_ForecastingModelMeta):
    """
    Abstract base class for forecasting models.

    Derived classes must implement the :meth:`model` method.
    """
    @abstractmethod
    def model(self, zero_data, covariates):
        """
        Generative model definition.

        Implementations must call the :meth:`predict` method exactly once.

        Implementations must draw all time-dependent noise inside the
        :meth:`time_plate` .  The prediction passed to :meth:`predict` must be
        a deterministic function of noise tensors that are independent over
        time. This requirement is slightly more general than state space
        models.

        :param zero_data: A zero tensor like the input data, but extended to
            the duration of the :meth:`time_plate` . This allows models to
            depend on the shape and device of data but not its value.
        :type zero_data: ~torch.Tensor
        :param covariates: A tensor of covariates with time dimension -2.
        :type covariates: ~torch.Tensor
        :returns: Return value is ignored.
        """
        raise NotImplementedError

    @property
    def time_plate(self):
        """
        :returns: A plate named "time" with size ``covariates.size(-2)`` and
            ``dim=-1``. This is available only during model execution.
        :rtype: :class:`~pyro.plate`
        """
        assert self._time_plate is not None, ".time_plate accessed outside of .model()"
        return self._time_plate

    def predict(self, noise_dist, prediction):
        """
        Prediction function.

        This requires the following unusual shape requirements:

        - ``noise_dist.event_shape == predction.shape[-2:]``
        - ``noise_dist.batch_shape == prediction.shape[:-2] + (1,)``,
          except during forecasting when ``noise_dist`` includes
          ``sample_shape`` and ``prediction`` may or may not include
          ``sample_shape``.
        """
        assert self._data is not None, ".predict() called outside .model()"
        assert self._forecast is None, ".predict() called twice"
        assert isinstance(noise_dist, dist.Distribution)
        assert isinstance(prediction, torch.Tensor)
        assert noise_dist.batch_shape[-1] == 1
        assert noise_dist.event_shape == prediction.shape[-2:]

        data = self._data
        t_obs = data.size(-2)
        t_cov = prediction.size(-2)
        if t_obs == t_cov:  # training
            pyro.sample("residual", noise_dist,
                        obs=(data - prediction).unsqueeze(-3))
            self._forecast = data.new_zeros(data.shape[:-2] + (0,) + data.shape[-1:])
        else:  # forecasting
            left_pred = prediction[..., :t_obs, :]
            right_pred = prediction[..., t_obs:, :]
            noise_dist = prefix_condition(noise_dist,
                                          data=(data - left_pred).unsqueeze(-3))
            noise = pyro.sample("residual", noise_dist).squeeze(-3)
            assert noise.shape[-data.dim():] == right_pred.shape[-data.dim():]
            self._forecast = right_pred + noise

    def forward(self, data, covariates):
        assert data.dim() >= 2
        assert covariates.dim() >= 2
        t_obs = data.size(-2)
        t_cov = covariates.size(-2)
        assert t_obs <= t_cov

        try:
            self._data = data
            self._time_plate = pyro.plate("time", t_cov, dim=-1)
            if t_obs == t_cov:  # training
                zero_data = data.new_zeros(()).expand(data.shape)
            else:  # forecasting
                zero_data = data.new_zeros(()).expand(
                    data.shape[:-2] + covariates.shape[-2:-1] + data.shape[-1:])
            self._forecast = None

            self.model(zero_data, covariates)

            assert self._forecast is not None, ".predict() was not called by .model()"
            return self._forecast
        finally:
            self._data = None
            self._time_plate = None
            self._forecast = None


class Forecaster(nn.Module):
    """
    Forecaster for a :class:`ForecastingModel` .

    On initialization, this fits a distribution using variational inference
    over latent variables and exact inference over the noise distribution,
    typically a :class:`~pyro.distributions.GaussianHMM` or variant.

    After construction this can be called to generate sample forecasts.

    :param ForecastingModel model: A forecasting model subclass instance.
    :param data: A tensor dataset with time dimension -2.
    :type data: ~torch.Tensor
    :param covariates: A tensor of covariates with time dimension -2.
        For models not using covariates, pass a shaped empty tensor
        ``torch.empty(duration, 0)``.
    :type covariates: ~torch.Tensor
    """
    def __init__(self, model, data, covariates, *,
                 learning_rate=0.01,
                 betas=(0.9, 0.99),
                 learning_rate_decay=0.1,
                 num_steps=1001,
                 log_every=100,
                 init_scale=0.1):
        assert data.size(-2) == covariates.size(-2)
        super().__init__()
        self.model = model
        self.guide = AutoNormal(self.model, init_loc_fn=init_to_sample, init_scale=init_scale)
        optim = ClippedAdam({"lr": learning_rate, "betas": betas,
                             "lrd": learning_rate_decay ** (1 / num_steps)})
        elbo = Trace_ELBO()
        elbo._guess_max_plate_nesting(self.model, self.guide, (data, covariates), {})
        elbo.max_plate_nesting = max(elbo.max_plate_nesting, 1)  # force a time plate

        svi = SVI(self.model, self.guide, optim, elbo)

        losses = []
        for step in range(num_steps):
            loss = svi.step(data, covariates) / data.numel()
            if log_every and step % log_every == 0:
                logger.info("step {: >4d} loss = {:0.6g}".format(step, loss))
            losses.append(loss)

        self.max_plate_nesting = elbo.max_plate_nesting
        self.losses = losses

    @torch.no_grad()
    def forward(self, data, covariates, num_samples):
        assert data.size(-2) < covariates.size(-2)
        assert self.max_plate_nesting >= 1
        dim = -1 - self.max_plate_nesting

        with poutine.trace() as tr:
            with pyro.plate("particles", num_samples, dim=dim):
                self.guide()
        with PrefixReplayMessenger(tr.trace):
            with pyro.plate("particles", num_samples, dim=dim):
                return self.model(data, covariates)
