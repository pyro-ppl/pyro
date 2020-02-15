# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import abstractmethod

import torch
import torch.nn as nn

import pyro
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal, init_to_sample
from pyro.nn.module import PyroModule
from pyro.optim import ClippedAdam

from .prefix import PrefixReplayMessenger, prefix_condition

logger = logging.getLogger(__name__)


class ForecastingModel(PyroModule):
    """
    Abstract base class for forecasting models.

    Derived classes should implement methods :meth:`get_locals` ,
    :meth:`get_obs_dist` , and optionally :meth:`get_globals` .
    """
    def get_globals(self, covariates):
        return None

    @abstractmethod
    def get_locals(self, covariates, gloabls):
        raise NotImplementedError

    @abstractmethod
    def get_noise_dist(self, covariates, gloabls):
        raise NotImplementedError

    @torch.no_grad()
    def forward(self, data, covariates):
        t_obs = data.size(-2)
        t_cov = covariates.size(-2)
        assert t_obs <= t_cov

        globals_ = self.get_globals(covariates)
        with pyro.plate("time", covariates.size(-2), dim=-1):
            prediction, locals_ = self.get_locals(covariates, globals_)
        noise_dist = self.get_noise_dist(covariates, globals_, locals_)

        if t_obs == t_cov:  # training
            pyro.sample("noise", noise_dist, obs=data - prediction)
        else:  # forecasting
            left_pred = prediction[..., :t_obs, :]
            right_pred = prediction[..., t_obs:, :]
            noise_dist = prefix_condition(noise_dist, data - left_pred)
            noise = pyro.sample("noise", noise_dist)
            return right_pred + noise

    def fit(self, data, covariates, **kwargs):
        """
        Convenience method to create and train a :class:`Forecaster` instance.
        """
        return Forecaster(self, data, covariates, **kwargs)


class Forecaster(nn.Module):
    """
    Forecaster for a :class:`ForecastingModel` .

    This fits a distribution using variational inference over latent variables
    and exact inference over the noise distribution, typically a
    :class:`~pyro.distributions.GaussianHMM` or variant.
    """
    def __init__(self, model, data, covariates, *,
                 learning_rate=0.01,
                 betas=(0.9, 0.99),
                 learning_rate_decay=0.1,
                 num_steps=1000,
                 init_scale=0.1):
        super().__init__()
        self.model = model
        assert data.size(-2) == covariates.size(-2)

        self.guide = AutoNormal(self.model, init_loc_fn=init_to_sample, init_scale=init_scale)
        optim = ClippedAdam({"lr": learning_rate, "betas": betas,
                             "lrd": learning_rate_decay ** (1 / num_steps)})
        elbo = Trace_ELBO()
        svi = SVI(self.model, self.guide, optim, elbo)

        self.losses = []
        for step in range(num_steps):
            loss = svi.step(data, covariates) / data.numel()
            self.losses.append(loss)
        self.max_plate_nesting = elbo.max_plate_nesting

    def forward(self, data, covariates, num_samples):
        assert data.size(-2) < covariates.size(-2)
        dim = -1 - self.max_plate_nesting
        with poutine.trace() as tr:
            with poutine.plate("particles", num_samples, dim=dim):
                self.guide()
        with PrefixReplayMessenger(tr.trace):
            with poutine.plate("particles", num_samples, dim=dim):
                return self.model(data, covariates)
