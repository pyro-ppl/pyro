# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABCMeta, abstractmethod
from contextlib import ExitStack

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal, init_to_sample
from pyro.infer.predictive import _guess_max_plate_nesting
from pyro.nn.module import PyroModule
from pyro.optim import DCTAdam

from .util import (MarkDCTParamMessenger, PrefixConditionMessenger, PrefixReplayMessenger, PrefixWarmStartMessenger,
                   reshape_batch, time_reparam_dct, time_reparam_haar)

logger = logging.getLogger(__name__)


class _ForecastingModelMeta(type(PyroModule), ABCMeta):
    pass


class ForecastingModel(PyroModule, metaclass=_ForecastingModelMeta):
    """
    Abstract base class for forecasting models.

    Derived classes must implement the :meth:`model` method.
    """
    def __init__(self):
        super().__init__()
        self._prefix_condition_data = {}

    @abstractmethod
    def model(self, zero_data, covariates):
        """
        Generative model definition.

        Implementations must call the :meth:`predict` method exactly once.

        Implementations must draw all time-dependent noise inside the
        :meth:`time_plate`. The prediction passed to :meth:`predict` must be a
        deterministic function of noise tensors that are independent over time.
        This requirement is slightly more general than state space models.

        :param zero_data: A zero tensor like the input data, but extended to
            the duration of the :meth:`time_plate`. This allows models to
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
        Prediction function, to be called by :meth:`model` implementations.

        This should be called outside of the  :meth:`time_plate`.

        This is similar to an observe statement in Pyro::

            pyro.sample("residual", noise_dist,
                        obs=(data - prediction))

        but with (1) additional reshaping logic to allow time-dependent
        ``noise_dist`` (most often a :class:`~pyro.distributions.GaussianHMM`
        or variant); and (2) additional logic to allow only a partial
        observation and forecast the remaining data.

        :param noise_dist: A noise distribution with ``.event_dim in {0,1,2}``.
            ``noise_dist`` is typically zero-mean or zero-median or zero-mode
            or somehow centered.
        :type noise_dist: ~pyro.distributions.Distribution
        :param prediction: A prediction for the data. This should have the same
            shape as ``data``, but broadcastable to full duration of the
            ``covariates``.
        :type prediction: ~torch.Tensor
        """
        assert self._data is not None, ".predict() called outside .model()"
        assert self._forecast is None, ".predict() called twice"
        assert isinstance(noise_dist, dist.Distribution)
        assert isinstance(prediction, torch.Tensor)
        if noise_dist.event_dim == 0:
            if noise_dist.batch_shape[-2:] != prediction.shape[-2:]:
                noise_dist = noise_dist.expand(
                    noise_dist.batch_shape[:-2] + prediction.shape[-2:])
            noise_dist = noise_dist.to_event(2)
        elif noise_dist.event_dim == 1:
            if noise_dist.batch_shape[-1:] != prediction.shape[-2:-1]:
                noise_dist = noise_dist.expand(
                    noise_dist.batch_shape[:-1] + prediction.shape[-2:-1])
            noise_dist = noise_dist.to_event(1)
        assert noise_dist.event_dim == 2
        assert noise_dist.event_shape == prediction.shape[-2:]

        # The following reshaping logic is required to reconcile batch and
        # event shapes. This would be unnecessary if Pyro used name dimensions
        # internally, e.g. using Funsor.
        #
        #     batch_shape                    | event_shape
        #     -------------------------------+----------------
        #  1. sample_shape + shape + (time,) | (obs_dim,)
        #  2.           sample_shape + shape | (time, obs_dim)
        #  3.    sample_shape + shape + (1,) | (time, obs_dim)
        #
        # Parameters like noise_dist.loc typically have shape as in 1.  However
        # calling .to_event(1) will shift the shapes resulting in 2., where
        # sample_shape+shape will be misaligned with other batch shapes in the
        # trace. To fix this the following logic "unsqueezes" the distribution,
        # resulting in correctly aligned shapes 3. Note the "time" dimension is
        # effectively moved from a batch dimension to an event dimension.
        noise_dist = reshape_batch(noise_dist, noise_dist.batch_shape + (1,))
        data = pyro.subsample(self._data.unsqueeze(-3), event_dim=2)
        prediction = prediction.unsqueeze(-3)

        # Create a sample site.
        t_obs = data.size(-2)
        t_cov = prediction.size(-2)
        if t_obs == t_cov:  # training
            pyro.sample("residual", noise_dist, obs=data - prediction)
            self._forecast = data.new_zeros(data.shape[:-2] + (0,) + data.shape[-1:])
        else:  # forecasting
            left_pred = prediction[..., :t_obs, :]
            right_pred = prediction[..., t_obs:, :]

            # This prefix_condition indirection is needed to ensure that
            # PrefixConditionMessenger is handled outside of the .model() call.
            self._prefix_condition_data["residual"] = data - left_pred
            noise = pyro.sample("residual", noise_dist)
            del self._prefix_condition_data["residual"]

            assert noise.shape[-data.dim():] == right_pred.shape[-data.dim():]
            self._forecast = right_pred + noise

        # Move the "time" batch dim back to its original place.
        assert self._forecast.size(-3) == 1
        self._forecast = self._forecast.squeeze(-3)

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
    Forecaster for a :class:`ForecastingModel` using variational inference.

    On initialization, this fits a distribution using variational inference
    over latent variables and exact inference over the noise distribution,
    typically a :class:`~pyro.distributions.GaussianHMM` or variant.

    After construction this can be called to generate sample forecasts.

    :ivar list losses: A list of losses recorded during training, typically
        used to debug convergence. Defined by ``loss = -elbo / data.numel()``.

    :param ForecastingModel model: A forecasting model subclass instance.
    :param data: A tensor dataset with time dimension -2.
    :type data: ~torch.Tensor
    :param covariates: A tensor of covariates with time dimension -2.
        For models not using covariates, pass a shaped empty tensor
        ``torch.empty(duration, 0)``.
    :type covariates: ~torch.Tensor

    :param guide: Optional guide instance. Defaults to a
        :class:`~pyro.infer.autoguide.AutoNormal`.
    :type guide: ~pyro.nn.module.PyroModule
    :param callable init_loc_fn: A per-site initialization function for the
        :class:`~pyro.infer.autoguide.AutoNormal` guide. Defaults to
        :func:`~pyro.infer.autoguide.initialization.init_to_sample`. See
        :ref:`autoguide-initialization` section for available functions.
    :param float init_scale: Initial uncertainty scale of the
        :class:`~pyro.infer.autoguide.AutoNormal` guide.
    :param callable create_plates: An optional function to create plates for
        subsampling with the :class:`~pyro.infer.autoguide.AutoNormal` guide.
    :param optim: An optional Pyro optimizer. Defaults to a freshly constructed
        :class:`~pyro.optim.optim.DCTAdam`.
    :type optim: ~pyro.optim.optim.PyroOptim
    :param float learning_rate: Learning rate used by
        :class:`~pyro.optim.optim.DCTAdam`.
    :param tuple betas: Coefficients for running averages used by
        :class:`~pyro.optim.optim.DCTAdam`.
    :param float learning_rate_decay: Learning rate decay used by
        :class:`~pyro.optim.optim.DCTAdam`. Note this is the total decay
        over all ``num_steps``, not the per-step decay factor.
    :param float clip_norm: Norm used for gradient clipping during
        optimization. Defaults to 10.0.
    :param str time_reparam: If not None (default), reparameterize all
        time-dependent variables via the Haar wavelet transform (if "haar") or
        the discrete cosine transform (if "dct").
    :param bool dct_gradients: Whether to discrete cosine transform gradients
        in :class:`~pyro.optim.optim.DCTAdam`. Defaults to False.
    :param bool subsample_aware: whether to update gradient statistics only
        for those elements that appear in a subsample. This is used
        by :class:`~pyro.optim.optim.DCTAdam`.
    :param int num_steps: Number of :class:`~pyro.infer.svi.SVI` steps.
    :param int num_particles: Number of particles used to compute the
        :class:`~pyro.infer.elbo.ELBO`.
    :param bool vectorize_particles: If ``num_particles > 1``, determines
        whether to vectorize computation of the :class:`~pyro.infer.elbo.ELBO`.
        Defaults to True. Set to False for models with dynamic control flow.
    :param bool warm_start: Whether to warm start parameters from a smaller
        time window. Note this may introduce statistical leakage; usage is
        recommended for model exploration purposes only and should be disabled
        when publishing metrics.
    :param int log_every: Number of training steps between logging messages.
    """
    def __init__(self, model, data, covariates, *,
                 guide=None,
                 init_loc_fn=init_to_sample,
                 init_scale=0.1,
                 create_plates=None,
                 optim=None,
                 learning_rate=0.01,
                 betas=(0.9, 0.99),
                 learning_rate_decay=0.1,
                 clip_norm=10.0,
                 time_reparam=None,
                 dct_gradients=False,
                 subsample_aware=False,
                 num_steps=1001,
                 num_particles=1,
                 vectorize_particles=True,
                 warm_start=False,
                 log_every=100):
        assert data.size(-2) == covariates.size(-2)
        super().__init__()
        self.model = model
        if time_reparam == "haar":
            model = poutine.reparam(model, time_reparam_haar)
        elif time_reparam == "dct":
            model = poutine.reparam(model, time_reparam_dct)
        elif time_reparam is not None:
            raise ValueError("unknown time_reparam: {}".format(time_reparam))
        if guide is None:
            guide = AutoNormal(model, init_loc_fn=init_loc_fn, init_scale=init_scale,
                               create_plates=create_plates)
        self.guide = guide

        # Initialize.
        if warm_start:
            model = PrefixWarmStartMessenger()(model)
            guide = PrefixWarmStartMessenger()(guide)
        if dct_gradients:
            model = MarkDCTParamMessenger("time")(model)
            guide = MarkDCTParamMessenger("time")(guide)
        elbo = Trace_ELBO(num_particles=num_particles,
                          vectorize_particles=vectorize_particles)
        elbo._guess_max_plate_nesting(model, guide, (data, covariates), {})
        elbo.max_plate_nesting = max(elbo.max_plate_nesting, 1)  # force a time plate

        losses = []
        if num_steps:
            if optim is None:
                optim = DCTAdam({"lr": learning_rate, "betas": betas,
                                 "lrd": learning_rate_decay ** (1 / num_steps),
                                 "clip_norm": clip_norm,
                                 "subsample_aware": subsample_aware})
            svi = SVI(model, guide, optim, elbo)
            for step in range(num_steps):
                loss = svi.step(data, covariates) / data.numel()
                if log_every and step % log_every == 0:
                    logger.info("step {: >4d} loss = {:0.6g}".format(step, loss))
                losses.append(loss)

        self.guide.create_plates = None  # Disable subsampling after training.
        self.max_plate_nesting = elbo.max_plate_nesting
        self.losses = losses

    def __call__(self, data, covariates, num_samples, batch_size=None):
        """
        Samples forecasted values of data for time steps in ``[t1,t2)``, where
        ``t1 = data.size(-2)`` is the duration of observed data and ``t2 =
        covariates.size(-2)`` is the extended duration of covariates. For
        example to forecast 7 days forward conditioned on 30 days of
        observations, set ``t1=30`` and ``t2=37``.

        :param data: A tensor dataset with time dimension -2.
        :type data: ~torch.Tensor
        :param covariates: A tensor of covariates with time dimension -2.
            For models not using covariates, pass a shaped empty tensor
            ``torch.empty(duration, 0)``.
        :type covariates: ~torch.Tensor
        :param int num_samples: The number of samples to generate.
        :param int batch_size: Optional batch size for sampling. This is useful
            for generating many samples from models with large memory
            footprint. Defaults to ``num_samples``.
        :returns: A batch of joint posterior samples of shape
            ``(num_samples,1,...,1) + data.shape[:-2] + (t2-t1,data.size(-1))``,
            where the ``1``'s are inserted to avoid conflict with model plates.
        :rtype: ~torch.Tensor
        """
        return super().__call__(data, covariates, num_samples, batch_size)

    def forward(self, data, covariates, num_samples, batch_size=None):
        assert data.size(-2) <= covariates.size(-2)
        assert isinstance(num_samples, int) and num_samples > 0
        if batch_size is not None:
            batches = []
            while num_samples > 0:
                batch = self.forward(data, covariates, min(num_samples, batch_size))
                batches.append(batch)
                num_samples -= batch_size
            return torch.cat(batches)

        assert self.max_plate_nesting >= 1
        dim = -1 - self.max_plate_nesting

        with torch.no_grad():
            with poutine.block(), poutine.trace() as tr:
                with pyro.plate("particles", num_samples, dim=dim):
                    self.guide(data, covariates)
            with ExitStack() as stack:
                if data.size(-2) < covariates.size(-2):
                    stack.enter_context(PrefixReplayMessenger(tr.trace))
                    stack.enter_context(
                        PrefixConditionMessenger(self.model._prefix_condition_data))
                else:
                    stack.enter_context(poutine.replay(trace=tr.trace))
                with pyro.plate("particles", num_samples, dim=dim):
                    return self.model(data, covariates)


class HMCForecaster(nn.Module):
    """
    Forecaster for a :class:`ForecastingModel` using Hamiltonian Monte Carlo.

    On initialization, this will run :class:`~pyro.infer.mcmc.nuts.NUTS`
    sampler to get posterior samples of the model.

    After construction, this can be called to generate sample forecasts.

    :param ForecastingModel model: A forecasting model subclass instance.
    :param data: A tensor dataset with time dimension -2.
    :type data: ~torch.Tensor
    :param covariates: A tensor of covariates with time dimension -2.
        For models not using covariates, pass a shaped empty tensor
        ``torch.empty(duration, 0)``.
    :type covariates: ~torch.Tensor
    :param int num_warmup: number of MCMC warmup steps.
    :param int num_samples: number of MCMC samples.
    :param int num_chains: number of parallel MCMC chains.
    :param bool dense_mass: a flag to control whether the mass matrix is dense
        or diagonal. Defaults to False.
    :param str time_reparam: If not None (default), reparameterize all
        time-dependent variables via the Haar wavelet transform (if "haar") or
        the discrete cosine transform (if "dct").
    :param bool jit_compile: whether to use the PyTorch JIT to trace the log
        density computation, and use this optimized executable trace in the
        integrator. Defaults to False.
    :param int max_tree_depth: Max depth of the binary tree created during the
        doubling scheme of the :class:`~pyro.infer.mcmc.nuts.NUTS` sampler.
        Defaults to 10.
    """
    def __init__(self, model, data, covariates=None, *,
                 num_warmup=1000, num_samples=1000, num_chains=1, time_reparam=None,
                 dense_mass=False, jit_compile=False, max_tree_depth=10):
        assert data.size(-2) == covariates.size(-2)
        super().__init__()
        if time_reparam == "haar":
            model = poutine.reparam(model, time_reparam_haar)
        elif time_reparam == "dct":
            model = poutine.reparam(model, time_reparam_dct)
        elif time_reparam is not None:
            raise ValueError("unknown time_reparam: {}".format(time_reparam))
        self.model = model
        max_plate_nesting = _guess_max_plate_nesting(model, (data, covariates), {})
        self.max_plate_nesting = max(max_plate_nesting, 1)  # force a time plate

        kernel = NUTS(model, full_mass=dense_mass, jit_compile=jit_compile, ignore_jit_warnings=True,
                      max_tree_depth=max_tree_depth, max_plate_nesting=max_plate_nesting)
        mcmc = MCMC(kernel, warmup_steps=num_warmup, num_samples=num_samples, num_chains=num_chains)
        mcmc.run(data, covariates)
        # conditions to compute rhat
        if (num_chains == 1 and num_samples >= 4) or (num_chains > 1 and num_samples >= 2):
            mcmc.summary()

        # inspect the model with particles plate = 1, so that we can reshape samples to
        # add any missing plate dim in front.
        with poutine.trace() as tr:
            with pyro.plate("particles", 1, dim=-self.max_plate_nesting - 1):
                model(data, covariates)

        self._trace = tr.trace
        self._samples = mcmc.get_samples()
        self._num_samples = num_samples * num_chains
        for name, node in list(self._trace.nodes.items()):
            if name not in self._samples:
                del self._trace.nodes[name]

    def __call__(self, data, covariates, num_samples, batch_size=None):
        """
        Samples forecasted values of data for time steps in ``[t1,t2)``, where
        ``t1 = data.size(-2)`` is the duration of observed data and ``t2 =
        covariates.size(-2)`` is the extended duration of covariates. For
        example to forecast 7 days forward conditioned on 30 days of
        observations, set ``t1=30`` and ``t2=37``.

        :param data: A tensor dataset with time dimension -2.
        :type data: ~torch.Tensor
        :param covariates: A tensor of covariates with time dimension -2.
            For models not using covariates, pass a shaped empty tensor
            ``torch.empty(duration, 0)``.
        :type covariates: ~torch.Tensor
        :param int num_samples: The number of samples to generate.
        :param int batch_size: Optional batch size for sampling. This is useful
            for generating many samples from models with large memory
            footprint. Defaults to ``num_samples``.
        :returns: A batch of joint posterior samples of shape
            ``(num_samples,1,...,1) + data.shape[:-2] + (t2-t1,data.size(-1))``,
            where the ``1``'s are inserted to avoid conflict with model plates.
        :rtype: ~torch.Tensor
        """
        return super().__call__(data, covariates, num_samples, batch_size)

    def forward(self, data, covariates, num_samples, batch_size=None):
        assert data.size(-2) <= covariates.size(-2)
        assert isinstance(num_samples, int) and num_samples > 0
        if batch_size is not None:
            batches = []
            while num_samples > 0:
                batch = self.forward(data, covariates, min(num_samples, batch_size))
                batches.append(batch)
                num_samples -= batch_size
            return torch.cat(batches)

        assert self.max_plate_nesting >= 1
        dim = -1 - self.max_plate_nesting

        with torch.no_grad():
            weights = torch.ones(self._num_samples, device=data.device)
            indices = torch.multinomial(weights, num_samples, replacement=num_samples > self._num_samples)
            for name, node in list(self._trace.nodes.items()):
                sample = self._samples[name].index_select(0, indices)
                node['value'] = sample.reshape(
                    (num_samples,) + (1,) * (node['value'].dim() - sample.dim()) + sample.shape[1:])

            with ExitStack() as stack:
                if data.size(-2) < covariates.size(-2):
                    stack.enter_context(PrefixReplayMessenger(self._trace))
                    stack.enter_context(
                        PrefixConditionMessenger(self.model._prefix_condition_data))
                with pyro.plate("particles", num_samples, dim=dim):
                    return self.model(data, covariates)
