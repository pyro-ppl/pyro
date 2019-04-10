from __future__ import absolute_import, division, print_function

from collections import defaultdict

import pyro
import pyro.ops.jit
from pyro import poutine
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.infer.util import torch_item
from pyro.util import warn_if_nan


def _compute_mmd(X, Z, kernel):
    mmd = kernel(X).mean() + kernel(Z).mean() - 2 * kernel(X, Z).mean()
    return mmd


class Trace_MMD(Trace_ELBO):
    """
    An objective similar to ELBO, but with Maximum Mean Discrepancy (MMD)
    between marginal variational posterior and prior distributions
    instead of KL-divergence between variational posterior and prior distributions
    as in vanilla ELBO. See [1] for the corresponding variant of VAE model,
    which is a special case of InfoVAE model [2].

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

    def _differentiable_loss_parts(self, model, guide, *args, **kwargs):
        loglikelihood = 0.0
        penalty = 0.0
        model_trace_independent = poutine.trace(model, graph_type="flat").get_trace(*args, **kwargs)
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            loglikelihood_particle = 0.0
            penalty_particle = 0.0
            for name, model_site in model_trace.nodes.items():
                if model_site['type'] == 'sample':
                    if name in guide_trace:
                        guide_site = guide_trace.nodes[name]
                        independent_model_site = model_trace_independent.nodes[name]
                        model_samples = independent_model_site['value']
                        guide_samples = guide_site['value']
                        divergence = _compute_mmd(
                            model_samples.view(model_samples.size(0), -1),
                            guide_samples.view(guide_samples.size(0), -1),
                            kernel=self._kernel[name]
                        )
                        penalty_particle = penalty_particle + self._mmd_scale[name] * divergence
                    else:
                        loglikelihood_particle = loglikelihood_particle + model_site['log_prob_sum']
            loglikelihood = loglikelihood_particle / self.num_particles + loglikelihood
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
