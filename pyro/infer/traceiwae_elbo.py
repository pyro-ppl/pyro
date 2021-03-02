# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import pyro.poutine as poutine
from pyro.distributions.util import detach
from pyro.infer.elbo import ELBO
from pyro.infer.util import is_validation_enabled, torch_item
from pyro.poutine.replay_messenger import ReplayMessenger
from pyro.poutine.util import prune_subsample_sites
from pyro.util import (
    check_if_enumerated,
    check_model_guide_match,
    check_site_shape,
    warn_if_nan,
)


class DetachReplayMessenger(ReplayMessenger):
    def _pyro_sample(self, msg):
        super()._pyro_sample(msg)
        if msg["name"] in self.trace:
            msg["value"] = msg["value"].detach()


class TraceIWAE_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI using the doubly reparameterized
    gradient estimator [1].

    **References**

    [1] G. Tucker, D. Wawson, S. Gu, C.J. Maddison (2018)
        Doubly Reparameterized Gradient Estimators for Monte Carlo Objectives
        https://arxiv.org/abs/1810.04152
    """
    def _get_trace(*args, **kwargs):
        raise ValueError("Use see _get_importance_trace() instead")

    def _get_importance_trace(self, model, guide, args, kwargs):
        assert self.vectorize_particles
        if self.max_plate_nesting == math.inf:
            self._guess_max_plate_nesting(model, guide, args, kwargs)
        model = self._vectorized_num_particles(model)
        guide = self._vectorized_num_particles(guide)

        guide_trace = poutine.trace(guide).get_trace(*args, **kwargs)
        with poutine.trace() as tr:
            with DetachReplayMessenger(trace=guide_trace):
                model(*args, **kwargs)
        model_trace = tr.trace
        if is_validation_enabled():
            check_model_guide_match(model_trace, guide_trace, self.max_plate_nesting)

        guide_trace = prune_subsample_sites(guide_trace)
        model_trace = prune_subsample_sites(model_trace)

        # Note we can avoid computing log_prob and score_parts unless validating.
        if is_validation_enabled():
            model_trace.compute_log_prob()
            guide_trace.compute_log_prob()
            for site in model_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, self.max_plate_nesting)
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, self.max_plate_nesting)
            check_if_enumerated(guide_trace)

        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        loss = self.differentiable_loss(model, guide, *args, **kwargs)
        return torch_item(loss)
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        loss = self.differentiable_loss(model, guide, *args, **kwargs)
        loss.backward()
        return torch_item(loss)

    def differentiable_loss(self, model, guide, *args, **kwargs):
        model_trace, guide_trace = self._get_importance_trace(model, guide, args, kwargs)

        # The following computation follows Sec. 8.3 Eqn. (12) of [1].
        log_w_bar = 0.  # all gradients stopped
        log_w_hat = 0.  # gradients stopped wrt distribution parameters
        log_p_tilde = 0.  # gradients stopped wrt reparameterized z

        def particle_sum(x):
            "sum out everything but the particle plate dimension"
            assert x.size(0) == self.num_particles
            return x.reshape(self.num_particles, -1).sum(-1)

        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                fn = site["fn"]
                z_detach = site["value"]
                if name in guide_trace:
                    z = guide_trace.nodes[name]["value"]
                else:
                    z = z_detach

                log_p = particle_sum(detach(fn).log_prob(z))
                log_w_bar = log_w_bar + log_p.detach()
                log_w_hat = log_w_hat + log_p
                log_p_tilde = log_p_tilde + particle_sum(fn.log_prob(z_detach))

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                fn = site["fn"]
                z = site["value"]

                log_q = particle_sum(detach(fn).log_prob(z))
                log_w_bar = log_w_bar - log_q.detach()
                log_w_hat = log_w_hat - log_q

        log_W_bar = log_w_bar.logsumexp(0)
        weight_bar = (log_w_bar - log_W_bar).exp()
        surrogate_elbo = weight_bar.dot(log_p_tilde) + weight_bar.pow(2).dot(log_w_hat)
        elbo = log_W_bar - math.log(self.num_particles)
        loss = -elbo + surrogate_elbo.detach() - surrogate_elbo

        warn_if_nan(loss, "loss")
        return loss
