from __future__ import absolute_import, division, print_function

import math
import warnings

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import iter_discrete_traces
from pyro.infer.util import TensorTree
from pyro.poutine.enumerate_poutine import EnumeratePoutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, is_nan


def _compute_upstream_grads(trace):
    upstream_grads = TensorTree()

    for site in trace.nodes.values():
        if site["type"] != "sample":
            continue
        score_function_term = site["score_parts"].score_function
        if is_identically_zero(score_function_term):
            continue
        cond_indep_stack = tuple(site["cond_indep_stack"])
        upstream_grads.add(cond_indep_stack, score_function_term)

    return upstream_grads


class TraceEnum_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI that supports enumeration
    over discrete sample sites.

    This implementation makes strong restrictions on the dependency
    structure of the ``model`` and ``guide``:
    Across :func:`~pyro.irange` and :func:`~pyro.iarange` blocks,
    both dependency graphs should follow a tree structure. That is,
    no variable outside of a block can depend on a variable in the block.
    """

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a trace generator
        """
        # enable parallel enumeration
        guide = EnumeratePoutine(guide, first_available_dim=self.max_iarange_nesting)

        for i in range(self.num_particles):
            # iterate over a bag of traces, one trace per particle
            for weights, guide_trace in iter_discrete_traces("flat", guide, *args, **kwargs):
                model_trace = poutine.trace(poutine.replay(model, guide_trace),
                                            graph_type="flat").get_trace(*args, **kwargs)

                check_model_guide_match(model_trace, guide_trace)
                guide_trace = prune_subsample_sites(guide_trace)
                model_trace = prune_subsample_sites(model_trace)

                model_trace.compute_batch_log_pdf()
                guide_trace.compute_score_parts()
                for site in model_trace.nodes.values():
                    if site["type"] == "sample":
                        check_site_shape(site, self.max_iarange_nesting)
                for site in guide_trace.nodes.values():
                    if site["type"] == "sample":
                        check_site_shape(site, self.max_iarange_nesting)

                yield weights, model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for weights, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = 0
            for name, model_site in model_trace.nodes.items():
                if model_site["type"] != "sample":
                    continue

                # grab weights introduced by enumeration
                cond_indep_stack = tuple(model_site["cond_indep_stack"])
                weight = weights.get_upstream(cond_indep_stack)
                if weight is None:
                    continue
                print('DEBUG {} weight = {}'.format(name, weight))

                model_log_pdf = model_site["batch_log_pdf"]
                if model_site["is_observed"]:
                    elbo_particle += (model_log_pdf * weight).sum().item()
                else:
                    guide_log_pdf = guide_trace.nodes[name]["batch_log_pdf"]
                    log_r = model_log_pdf - guide_log_pdf
                    print('DEBUG {} log_r = {}'.format(name, log_r))
                    elbo_particle += (log_r * weight).sum().item()

            elbo += elbo_particle / self.num_particles

        loss = -elbo
        if math.isnan(loss):
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
        for weights, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            upstream_grads = _compute_upstream_grads(guide_trace)
            elbo_particle = 0
            surrogate_elbo_particle = 0
            # compute elbo and surrogate elbo
            for name, model_site in model_trace.nodes.items():
                if model_site["type"] != "sample":
                    continue

                # grab weights introduced by enumeration
                cond_indep_stack = tuple(model_site["cond_indep_stack"])
                weight = weights.get_upstream(cond_indep_stack)
                if weight is None:
                    continue

                model_log_pdf = model_site["batch_log_pdf"]
                if model_site["is_observed"]:
                    model_log_pdf_sum = (model_site["batch_log_pdf"] * weight).sum()
                    elbo_particle += model_log_pdf_sum.item()
                    surrogate_elbo_particle = surrogate_elbo_particle + model_log_pdf_sum
                else:
                    guide_log_pdf, _, entropy_term = guide_trace.nodes[name]["score_parts"]
                    score_function_term = upstream_grads.get_upstream(cond_indep_stack)
                    log_r = model_log_pdf - guide_log_pdf
                    surrogate_elbo_site = model_log_pdf

                    if not is_identically_zero(entropy_term):
                        surrogate_elbo_site = surrogate_elbo_site - entropy_term

                    if score_function_term is not None:
                        surrogate_elbo_site = surrogate_elbo_site + log_r.detach() * score_function_term

                    elbo_particle += (log_r * weight).sum().item()
                    surrogate_elbo_particle = surrogate_elbo_particle + (surrogate_elbo_site * weight).sum()

            elbo += elbo_particle / self.num_particles

            trainable_params = set(site["value"]
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
