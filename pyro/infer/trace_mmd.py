# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict

import torch

import pyro
import pyro.ops.jit
from pyro import poutine
from pyro.infer.elbo import ELBO
from pyro.infer.util import torch_item, is_validation_enabled
from pyro.infer.enum import get_importance_trace
from pyro.util import check_if_enumerated, warn_if_nan


def _compute_mmd(X, Z, kernel):
    mmd = torch.mean(kernel(X)) + torch.mean(kernel(Z)) - torch.mean(kernel(X, Z)) * 2
    return mmd


class Trace_MMD(ELBO):
    """
    An objective similar to ELBO, but with Maximum Mean Discrepancy (MMD)
    between marginal variational posterior and prior distributions
    instead of KL-divergence between variational posterior and prior distributions
    as in vanilla ELBO.
    The simplest example is MMD-VAE model [1]. The corresponding loss function is given as follows:

        :math: `L(\\theta, \\phi) = -E_{p_{data}(x)} E_{q(z | x; \\phi)} \\log p(x | z; \\theta) +
        MMD(q(z; \\phi) \\| p(z))`,

    where z is a latent code. MMD between two distributions is defined as follows:

        :math: `MMD(q(z) \\| p(z)) = E_{p(z), p(z')} k(z,z') + E_{q(z), q(z')} k(z,z') - 2 E_{p(z), q(z')} k(z,z')`,

    where k is a kernel.

    DISCLAIMER: this implementation treats only the particle dimension as batch dimension when computing MMD.
    All other dimensions are treated as event dimensions.
    For this reason, one needs large `num_particles` in order to have reasonable variance of MMD Monte-Carlo estimate.
    As a consequence, it is recommended to set `vectorize_particles=True` (default).
    The general case will be implemented in future versions.

    :param kernel: A kernel used to compute MMD.
        An instance of :class: `pyro.contrib.gp.kernels.kernel.Kernel`,
        or a dict that maps latent variable names to instances of :class: `pyro.contrib.gp.kernels.kernel.Kernel`.
        In the latter case, different kernels are used for different latent variables.

    :param mmd_scale: A scaling factor for MMD terms.
        Float, or a dict that maps latent variable names to floats.
        In the latter case, different scaling factors are used for different latent variables.

    References

    [1] `A Tutorial on Information Maximizing Variational Autoencoders (InfoVAE)`
        Shengjia Zhao
        https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/

    [2] `InfoVAE: Balancing Learning and Inference in Variational Autoencoders`
        Shengjia Zhao, Jiaming Song, Stefano Ermon
    """

    def __init__(self,
                 kernel, mmd_scale=1,
                 num_particles=10,
                 max_plate_nesting=float('inf'),
                 max_iarange_nesting=None,  # DEPRECATED
                 vectorize_particles=True,
                 strict_enumeration_warning=True,
                 ignore_jit_warnings=False,
                 retain_graph=None):
        super().__init__(
            num_particles, max_plate_nesting, max_iarange_nesting, vectorize_particles,
            strict_enumeration_warning, ignore_jit_warnings, retain_graph,
        )
        self._kernel = None
        self._mmd_scale = None
        self.kernel = kernel
        self.mmd_scale = mmd_scale

    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, kernel):
        if isinstance(kernel, dict):
            # fix kernel's parameters
            for k in kernel.values():
                if isinstance(k, pyro.contrib.gp.kernels.kernel.Kernel):
                    k.requires_grad_(False)
                else:
                    raise TypeError("`kernel` values should be instances of `pyro.contrib.gp.kernels.kernel.Kernel`")
            self._kernel = kernel
        elif isinstance(kernel, pyro.contrib.gp.kernels.kernel.Kernel):
            kernel.requires_grad_(False)
            self._kernel = defaultdict(lambda: kernel)
        else:
            raise TypeError("`kernel` should be an instance of `pyro.contrib.gp.kernels.kernel.Kernel`")

    @property
    def mmd_scale(self):
        return self._mmd_scale

    @mmd_scale.setter
    def mmd_scale(self, mmd_scale):
        if isinstance(mmd_scale, dict):
            self._mmd_scale = mmd_scale
        elif isinstance(mmd_scale, (int, float)):
            self._mmd_scale = defaultdict(lambda: float(mmd_scale))
        else:
            raise TypeError("`mmd_scale` should be either float, or a dict of floats")

    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def _differentiable_loss_parts(self, model, guide, args, kwargs):
        all_model_samples = defaultdict(list)
        all_guide_samples = defaultdict(list)

        loglikelihood = 0.0
        penalty = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            if self.vectorize_particles:
                model_trace_independent = poutine.trace(
                    self._vectorized_num_particles(model)
                ).get_trace(*args, **kwargs)
            else:
                model_trace_independent = poutine.trace(model, graph_type='flat').get_trace(*args, **kwargs)

            loglikelihood_particle = 0.0
            for name, model_site in model_trace.nodes.items():
                if model_site['type'] == 'sample':
                    if name in guide_trace and not model_site['is_observed']:
                        guide_site = guide_trace.nodes[name]
                        independent_model_site = model_trace_independent.nodes[name]
                        if not independent_model_site["fn"].has_rsample:
                            raise ValueError("Model site {} is not reparameterizable".format(name))
                        if not guide_site["fn"].has_rsample:
                            raise ValueError("Guide site {} is not reparameterizable".format(name))

                        particle_dim = -self.max_plate_nesting - independent_model_site["fn"].event_dim

                        model_samples = independent_model_site['value']
                        guide_samples = guide_site['value']

                        if self.vectorize_particles:
                            model_samples = model_samples.transpose(-model_samples.dim(), particle_dim)
                            model_samples = model_samples.view(model_samples.shape[0], -1)

                            guide_samples = guide_samples.transpose(-guide_samples.dim(), particle_dim)
                            guide_samples = guide_samples.view(guide_samples.shape[0], -1)
                        else:
                            model_samples = model_samples.view(1, -1)
                            guide_samples = guide_samples.view(1, -1)

                        all_model_samples[name].append(model_samples)
                        all_guide_samples[name].append(guide_samples)
                    else:
                        loglikelihood_particle = loglikelihood_particle + model_site['log_prob_sum']

            loglikelihood = loglikelihood_particle / self.num_particles + loglikelihood

        for name in all_model_samples.keys():
            all_model_samples[name] = torch.cat(all_model_samples[name])
            all_guide_samples[name] = torch.cat(all_guide_samples[name])
            divergence = _compute_mmd(all_model_samples[name], all_guide_samples[name], kernel=self._kernel[name])
            penalty = self._mmd_scale[name] * divergence + penalty

        warn_if_nan(loglikelihood, "loglikelihood")
        warn_if_nan(penalty, "penalty")
        return loglikelihood, penalty

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        Computes the MMD-VAE-type loss [1]. Calling backward on the latter
        leads to valid gradient estimates as long as latent variables
        in both the guide and the model are reparameterizable.

        References

        [1] `A Tutorial on Information Maximizing Variational Autoencoders (InfoVAE)`
            Shengjia Zhao
            https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
        """
        loglikelihood, penalty = self._differentiable_loss_parts(model, guide, args, kwargs)
        loss = -loglikelihood + penalty
        warn_if_nan(loss, "loss")
        return loss

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the MMD-VAE-type loss [1]
        :rtype: float

        Computes the MMD-VAE-type loss with an estimator that uses num_particles many samples/particles.

        References

        [1] `A Tutorial on Information Maximizing Variational Autoencoders (InfoVAE)`
            Shengjia Zhao
            https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
        """
        loss = self.differentiable_loss(model, guide, *args, **kwargs)
        return torch_item(loss)

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the MMD-VAE-type loss [1]
        :rtype: float

        Computes the MMD-VAE-type loss and performs backward on it.
        Leads to valid gradient estimates as long as latent variables
        in both the guide and the model are reparameterizable.
        Num_particles many samples are used to form the estimators.

        References

        [1] `A Tutorial on Information Maximizing Variational Autoencoders (InfoVAE)`
            Shengjia Zhao
            https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
        """
        loss = self.differentiable_loss(model, guide, *args, **kwargs)
        loss.backward(retain_graph=self.retain_graph)
        return torch_item(loss)
