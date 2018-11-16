from __future__ import absolute_import, division, print_function

from torch.distributions import kl_divergence

import pyro
from pyro.distributions.util import is_identically_zero
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.infer.util import torch_item
from pyro.util import warn_if_nan


class Trace_MeanFieldELBO(Trace_ELBO):
    """
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide. The gradient estimator includes
    partial Rao-Blackwellization for reducing the variance of the estimator when
    non-reparameterizable random variables are present. The Rao-Blackwellization is
    partial in that it only uses conditional independence information that is marked
    by :class:`~pyro.plate` contexts. For more fine-grained Rao-Blackwellization,
    see :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`.

    References

    [1] Automated Variational Inference in Probabilistic Programming,
        David Wingate, Theo Weber

    [2] Black Box Variational Inference,
        Rajesh Ranganath, Sean Gerrish, David M. Blei
    """

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle, _ = self._differentiable_loss_particle(model_trace, guide_trace)
            elbo += elbo_particle / self.num_particles

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        elbo_particle = 0

        # compute elbo and surrogate elbo
        for name, model_site in model_trace.nodes.items():
            if model_site["type"] == "sample":
                if model_site["is_observed"]:
                    elbo_particle = elbo_particle + model_site["log_prob_sum"]
                else:
                    guide_site = guide_trace.nodes[name]
                    try:
                        kl_qp = kl_divergence(guide_site["fn"], model_site["fn"]).sum()
			elbo_particle = elbo_particle - kl_qp
                    except NotImplementedError:
			log_prob, score_function_term, entropy_term = guide_site["score_parts"]

			assert not is_identically_zero(entropy_term), \
			    "All distributions must be fully reparameterized"
			assert is_identically_zero(score_function_term), \
			    "All distributions must be fully reparameterized"

			elbo_particle = (elbo_particle + model_site["log_prob_sum"]
                                                   - entropy_term.sum())

        return -torch_item(elbo_particle), -elbo_particle
