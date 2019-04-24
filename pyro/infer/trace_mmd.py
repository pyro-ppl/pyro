from __future__ import absolute_import, division, print_function

from collections import defaultdict

import torch

import pyro
import pyro.ops.jit
from pyro import poutine
from pyro.infer.elbo import ELBO
from pyro.infer.util import torch_item, is_validation_enabled
from pyro.infer.enum import get_importance_trace
from pyro.util import check_if_enumerated, warn_if_nan


def _reshape_covariance_matrix(cov_matrix, num_particles, vectorize_particles):
    if vectorize_particles:
        return cov_matrix.view(
            num_particles, cov_matrix.shape[0] // num_particles,
            num_particles, cov_matrix.shape[1] // num_particles
        )
    else:
        return cov_matrix.view(
            1, cov_matrix.shape[0], 1, cov_matrix.shape[1]
        )


def _covariance_matrix_mean(cov_matrix, num_particles, vectorize_particles):
    cov_matrix = _reshape_covariance_matrix(cov_matrix, num_particles, vectorize_particles).transpose(2, 1)
    cov_matrix_mean = torch.mean(cov_matrix, [2, 3])
    return torch.diag(cov_matrix_mean).mean()


def _compute_mmd(X, Z, kernel, num_particles, vectorize_particles):
    mmd = _covariance_matrix_mean(kernel(X), num_particles, vectorize_particles) + \
          _covariance_matrix_mean(kernel(Z), num_particles, vectorize_particles) - \
          _covariance_matrix_mean(kernel(X, Z), num_particles, vectorize_particles) * 2
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
                 num_particles=1,
                 max_plate_nesting=float('inf'),
                 max_iarange_nesting=None,  # DEPRECATED
                 vectorize_particles=False,
                 strict_enumeration_warning=True,
                 ignore_jit_warnings=False,
                 retain_graph=None):
        super(Trace_MMD, self).__init__(
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
            self._kernel = kernel
        elif isinstance(kernel, pyro.contrib.gp.kernels.kernel.Kernel):
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

    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, *args, **kwargs)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def _differentiable_loss_parts(self, model, guide, *args, **kwargs):
        loglikelihood = 0.0
        penalty = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            if self.vectorize_particles:
                model_trace_independent = poutine.trace(
                    self._vectorized_num_particles(model)
                ).get_trace(*args, **kwargs)
            else:
                model_trace_independent = poutine.trace(model, graph_type='flat').get_trace(*args, **kwargs)
            loglikelihood_particle = 0.0
            penalty_particle = 0.0
            for name, model_site in model_trace.nodes.items():
                if model_site['type'] == 'sample':
                    if name in guide_trace and not model_site['is_observed']:
                        guide_site = guide_trace.nodes[name]
                        independent_model_site = model_trace_independent.nodes[name]
                        if not independent_model_site["fn"].has_rsample:
                            raise ValueError("Model site {} is not reparameterizable".format(name))
                        if not guide_site["fn"].has_rsample:
                            raise ValueError("Guide site {} is not reparameterizable".format(name))
                        model_samples = independent_model_site['value']
                        guide_samples = guide_site['value']
                        model_samples = model_samples.view(
                            -1, *[model_samples.size(j) for j in range(-independent_model_site['fn'].event_dim, 0)]
                        )
                        guide_samples = guide_samples.view(
                            -1, *[guide_samples.size(j) for j in range(-guide_site['fn'].event_dim, 0)]
                        )
                        divergence = _compute_mmd(
                            model_samples, guide_samples, kernel=self._kernel[name],
                            num_particles=self.num_particles, vectorize_particles=self.vectorize_particles
                        )
                        penalty_particle = penalty_particle + self._mmd_scale[name] * divergence
                    else:
                        loglikelihood_particle = loglikelihood_particle + model_site['log_prob_sum']
            loglikelihood = loglikelihood_particle / self.num_particles + loglikelihood
            if self.vectorize_particles:
                penalty = penalty_particle + penalty
            else:
                penalty = penalty_particle / self.num_particles + penalty

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
        loglikelihood, penalty = self._differentiable_loss_parts(model, guide, *args, **kwargs)
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
