from __future__ import absolute_import, division, print_function

import warnings

import pyro
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import iter_importance_traces
from pyro.infer.util import MultiFrameTensor, get_iarange_stacks
from pyro.util import is_nan


def _compute_log_r(model_trace, guide_trace):
    log_r = MultiFrameTensor()
    stacks = get_iarange_stacks(model_trace)
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            log_r_term = model_site["log_prob"]
            if not model_site["is_observed"]:
                log_r_term = log_r_term - guide_trace.nodes[name]["log_prob"]
            log_r.add((stacks[name], log_r_term.detach()))
    return log_r


class Trace_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide. The gradient estimator includes
    partial Rao-Blackwellization for reducing the variance of the estimator when
    non-reparameterizable random variables are present. The Rao-Blackwellization is
    partial in that it only uses conditional independence information that is marked
    by :class:`~pyro.iarange` contexts. For more fine-grained Rao-Blackwellization,
    see :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`.

    References

    [1] Automated Variational Inference in Probabilistic Programming,
        David Wingate, Theo Weber

    [2] Black Box Variational Inference,
        Rajesh Ranganath, Sean Gerrish, David M. Blei
    """

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a trace generator
        """
        return iter_importance_traces(num_particles=self.num_particles,
                                      graph_type="flat")(model, guide, *args, **kwargs)

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for _, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = (model_trace.log_pdf() - guide_trace.log_pdf()).item()
            elbo += elbo_particle / self.num_particles

        loss = -elbo
        if is_nan(loss):
            warnings.warn('Encountered NAN loss')
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        elbo = 0.0
        # grab a trace from the generator
        for _, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0
            surrogate_elbo_particle = 0
            log_r = None

            # compute elbo and surrogate elbo
            for name, model_site in model_trace.nodes.items():
                if model_site["type"] == "sample":
                    model_log_prob_sum = model_site["log_prob_sum"]
                    if model_site["is_observed"]:
                        elbo_particle = elbo_particle + model_log_prob_sum.item()
                        surrogate_elbo_particle = surrogate_elbo_particle + model_log_prob_sum
                    else:
                        guide_site = guide_trace.nodes[name]
                        guide_log_prob, score_function_term, entropy_term = guide_site["score_parts"]

                        elbo_particle = elbo_particle + (model_log_prob_sum.item() - guide_log_prob.sum().item())
                        surrogate_elbo_particle = surrogate_elbo_particle + model_log_prob_sum

                        if not is_identically_zero(entropy_term):
                            surrogate_elbo_particle -= entropy_term.sum()

                        if not is_identically_zero(score_function_term):
                            if log_r is None:
                                log_r = _compute_log_r(model_trace, guide_trace)
                            log_r_site = log_r.sum_to(guide_site["cond_indep_stack"])
                            surrogate_elbo_particle = surrogate_elbo_particle + (log_r_site * score_function_term).sum()

            elbo += elbo_particle / self.num_particles

            # collect parameters to train from model and guide
            trainable_params = set(site["value"].unconstrained()
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values()
                                   if site["type"] == "param")

            if trainable_params and getattr(surrogate_elbo_particle, 'requires_grad', False):
                surrogate_loss_particle = -surrogate_elbo_particle / self.num_particles
                surrogate_loss_particle.backward()
                pyro.get_param_store().mark_params_active(trainable_params)

        loss = -elbo
        if is_nan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
