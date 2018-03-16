from __future__ import absolute_import, division, print_function

import math
import warnings

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import iter_discrete_traces
from pyro.poutine.enumerate_poutine import EnumeratePoutine
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, is_nan


def _dict_iadd(items, key, value):
    if key in items:
        items[key] = items[key] + value
    else:
        items[key] = value


class MultiFrameDice(object):
    def __init__(self, trace):
        log_denom = {}
        log_probs = {}

        for site in trace.nodes.values():
            if site["type"] != "sample":
                continue
            log_prob = site['score_parts'].score_function  # not scaled by subsamling
            if is_identically_zero(log_prob):
                continue
            context = frozenset(f for f in site["cond_indep_stack"] if f.vectorized)
            if site["infer"].get("enumerate"):
                if "_enum_total" in site["infer"]:
                    _dict_iadd(log_denom, context, math.log(site["infer"]["_enum_total"]))
            else:
                log_prob = log_prob - log_prob.detach()
            _dict_iadd(log_probs, context, log_prob)

        self.log_denom = log_denom
        self.log_probs = log_probs

    def in_context(self, cond_indep_stack):
        target_context = frozenset(f for f in cond_indep_stack if f.vectorized)
        log_prob = 0
        for context, term in self.log_denom.items():
            if not context <= target_context:
                log_prob = log_prob - term
        for context, term in self.log_probs.items():
            if context <= target_context:
                log_prob = log_prob + term
        return 1 if is_identically_zero(log_prob) else log_prob.exp()


def _compute_dice_elbo(model_trace, guide_trace):
    dice = MultiFrameDice(guide_trace)
    elbo = 0
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            cost = model_site["batch_log_pdf"]
            if not model_site["is_observed"]:
                cost = cost - guide_trace.nodes[name]["batch_log_pdf"]
            dice_prob = dice.in_context(model_site["cond_indep_stack"])
            elbo = elbo + (dice_prob * cost).sum()
    return elbo


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
            for guide_trace in iter_discrete_traces("flat", guide, *args, **kwargs):
                model_trace = poutine.trace(poutine.replay(model, guide_trace),
                                            graph_type="flat").get_trace(*args, **kwargs)

                check_model_guide_match(model_trace, guide_trace)
                guide_trace = prune_subsample_sites(guide_trace)
                model_trace = prune_subsample_sites(model_trace)

                model_trace.compute_batch_log_pdf()
                for site in model_trace.nodes.values():
                    if site["type"] == "sample":
                        check_site_shape(site, self.max_iarange_nesting)
                guide_trace.compute_score_parts()
                for site in guide_trace.nodes.values():
                    if site["type"] == "sample":
                        check_site_shape(site, self.max_iarange_nesting)

                yield model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = _compute_dice_elbo(model_trace, guide_trace)
            if is_identically_zero(elbo_particle):
                continue

            elbo += elbo_particle.item() / self.num_particles

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
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = _compute_dice_elbo(model_trace, guide_trace)
            if is_identically_zero(elbo_particle):
                continue

            elbo += elbo_particle.item() / self.num_particles

            # collect parameters to train from model and guide
            trainable_params = set(site["value"]
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values()
                                   if site["type"] == "param")

            if trainable_params and elbo_particle.requires_grad:
                loss_particle = -elbo_particle
                (loss_particle / self.num_particles).backward()
                pyro.get_param_store().mark_params_active(trainable_params)

        loss = -elbo
        if is_nan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
