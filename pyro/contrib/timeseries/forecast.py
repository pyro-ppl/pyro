# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

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


class ForecastingModel(PyroModule):
    """
    Base class for forecasting models.

    Derived classes may override the :meth:`get_noise_dist` , and
    :meth:`get_globals` , and/or :meth:`get_locals` methods.

    :ivar list global_params: A list of the names of all global parameters.
    :ivar list local_params: A list of the names of all local parameters.
    :ivar list noise_params: A list of the names of all noise parameters.
    """
    def get_globals(self, zero_data, covariates):
        """
        Sample any time-independent global variables, e.g. trend parameters or
        noise scales.

        :param zero_data: A zero tensor like the input data. This allows models
            to depend on the shape and device of data but not its value.
        :type zero_data: ~torch.Tensor
        :param covariates: A tensor of covariates with time dimension -2.
        :type covariates: ~torch.Tensor
        :returns: Arbitrary global data to be passed to :meth:`get_locals` and
            :meth:`get_noise_dist` ; defaults to None.
        """
        return None

    def get_locals(self, zero_data, covariates, globals_):
        """
        Sample time-dependent parameters, e.g. error terms in a local trend.

        This will be called inside a time :class:`~pyro.primitives.plate` with
        dimension -1, ensuring all samples are independent over time.

        To sample a random time-dependent variable, you must factor that
        variable as a state-space model with independent noise drawn at each
        time step, and a final deterministic computation propagating the state.
        For example to predict brownian motion, you could define::

            def get_locals(self, zero_data, globals_):
                jumps = pyro.sample("jumps",
                                    dist.Normal(0, globals_["scale"])
                                        .expand(zero_data.size(-1))
                                        .to_event(1))
                prediction = torch.cumsum(jumps, dim=-2)
                locals_ = None
                return prediction, locals_

        :param zero_data: A zero tensor like the input data. This allows models
            to depend on the shape and device of data but not its value.
        :type zero_data: ~torch.Tensor
        :param covariates: A tensor of covariates with time dimension -2.
        :type covariates: ~torch.Tensor
        :param globals_: Any data computed by the :meth:`get_globals` method.
        :type globals_: object
        :returns: A pair ``(prediction, locals_)`` where ``prediction`` is a
            ``data``-shaped :class:`~torch.Tensor` (defaults to ``zero_data``)
            and ``locals_`` is arbitrary time-local data to be passed to
            :meth:`get_noise_dist` (defaults to None).
        :rtype tuple:
        """
        prediction = zero_data
        locals_ = None
        return prediction, locals_

    def get_noise_dist(self, zero_data, covariates, globals_, locals_):
        """
        Samples a noise distribution to be used as likelihood.

        :param zero_data: A zero tensor like the input data. This allows models
            to depend on the shape and device of data but not its value.
        :type zero_data: ~torch.Tensor
        :param covariates: A tensor of covariates with time dimension -2.
        :type covariates: ~torch.Tensor
        :param globals_: Any data computed by the :meth:`get_globals` method.
        :type globals_: object
        :param locals_: Any data computed by the :meth:`get_locals` method.
        :type locals_: object
        :returns: A distribution over noise, should treat time as an event
            dimension and satisfy ``event_shape == data.shape[-2:]``.
            Defaults to ``Normal(0,1)`` noise.
        :rtype: ~pyro.distributions.Distribution
        """
        return dist.Normal(zero_data, 1).to_event(zero_data.dim())

    @torch.no_grad()
    def forward(self, data, covariates):
        t_obs = data.size(-2)
        t_cov = covariates.size(-2)
        assert t_obs <= t_cov
        zero_data = data.new_zeros(()).expand(data.shape)

        # Globals.
        with poutine.trace(param_only=True) as tr:
            globals_ = self.get_globals(zero_data, covariates)
        self.global_params = tr.trace.param_nodes()

        # Locals.
        with poutine.trace(param_only=True) as tr:
            with pyro.plate("time", covariates.size(-2), dim=-1):
                prediction, locals_ = self.get_locals(zero_data, covariates, globals_)
        self.local_params = tr.trace.param_nodes()

        # Noise distribution.
        with poutine.trace(param_only=True) as tr:
            noise_dist = self.get_noise_dist(zero_data, covariates, globals_, locals_)
        self.noise_params = tr.trace.param_nodes()

        # Likelihood.
        with pyro.plate_stack("series", data.dim() - 2):
            if t_obs == t_cov:  # training
                pyro.sample("noise", noise_dist, obs=data - prediction)
            else:  # forecasting
                left_pred = prediction[..., :t_obs, :]
                right_pred = prediction[..., t_obs:, :]
                noise_dist = prefix_condition(noise_dist, data - left_pred)
                noise = pyro.sample("noise", noise_dist)
                return right_pred + noise


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
        svi = SVI(self.model, self.guide, optim, elbo)

        losses = []
        for step in range(num_steps):
            loss = svi.step(data, covariates) / data.numel()
            if log_every and step % log_every == 0:
                logging.info("step {: >4d} loss = {:0.6g}".format(step, loss))
            losses.append(loss)

        self.max_plate_nesting = elbo.max_plate_nesting
        self.losses = losses

    def forward(self, data, covariates, num_samples):
        assert data.size(-2) < covariates.size(-2)
        dim = -1 - self.max_plate_nesting
        with poutine.trace() as tr:
            with poutine.plate("particles", num_samples, dim=dim):
                self.guide()
        with PrefixReplayMessenger(tr.trace):
            with poutine.plate("particles", num_samples, dim=dim):
                return self.model(data, covariates)
