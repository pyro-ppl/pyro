from __future__ import absolute_import, division, print_function

import warnings
import weakref

import torch
from torch.distributions import kl_divergence

import pyro.ops.jit
from pyro.distributions.util import is_identically_zero, scale_and_mask
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.infer.util import is_validation_enabled, torch_item, _check_fully_reparametrized
from pyro.util import warn_if_nan


class TraceTailAdaptive_ELBO(Trace_ELBO):
    """

    References
    [1] "Variational Inference with Tail-adaptive f-Divergence", Dilin Wang,
        Hao Liu, Qiang Liu, NeurIPS 2018
        https://papers.nips.cc/paper/7816-variational-inference-with-tail-adaptive-f-divergence
    """
    def loss(self, model, guide, *args, **kwargs):
        """
        The tail-adaptive f-divergence in [1] is not straightforward to estimate;
        the gradients, however, can be estimated easily.
        """
        raise NotImplementedError("Loss method for TraceTailAdaptive_ELBO not implemented")

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        if not self.vectorize_particles:
            raise NotImplementedError("TraceTailAdaptive_ELBO only implemented for vectorize_particles==True")

        if self.num_particles == 1:
            warnings.warn("For num_particles==1 TraceTailAdaptive_ELBO uses the same loss function as Trace_ELBO. Increase num_particles to get an adaptive f-divergence.")

        log_p, log_q = 0, 0

        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                site_log_p = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                log_p = log_p + site_log_p


        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                site_log_q = site["log_prob"].reshape(self.num_particles, -1).sum(-1)
                log_q = log_q + site_log_q
                if is_validation_enabled():
                    _check_fully_reparametrized(site)

        log_pq = torch.logsumexp(log_p - log_q, dim=0)
        rank = torch.argsort(log_pq, descending=False)
        rank = torch.index_select(torch.arange(self.num_particles) + 1, -1, rank).float().type_as(log_pq)

        beta = -1.0
        gamma = torch.pow(rank, beta).detach()
        surrogate_loss = -(log_pq * gamma).sum() / gamma.sum()

        # we do not compute the loss, so return `inf`
        return float('inf'), surrogate_loss
