from __future__ import absolute_import, division, print_function

import warnings

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import iter_importance_traces
from pyro.infer.util import MultiFrameDice
from pyro.util import torch_isnan


def _compute_dice_elbo(model_trace, guide_trace):
    dice = MultiFrameDice(guide_trace)
    elbo = 0
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            cost = model_site["log_prob"]
            if not model_site["is_observed"]:
                cost = cost - guide_trace.nodes[name]["log_prob"]
            dice_prob = dice.in_context(model_site["cond_indep_stack"])
            # TODO use score_parts.entropy_term to "stick the landing"
            elbo = elbo + (dice_prob * cost).sum()
    return elbo


class TraceEnum_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI that supports enumeration
    over discrete sample sites.

    To enumerate over a sample site, the ``guide``'s sample site must specify
    either ``infer={'enumerate': 'sequential'}`` or
    ``infer={'enumerate': 'parallel'}``. To configure all sites at once, use
    :func:`~pyro.infer.enum.config_enumerate``.

    This assumes restricted dependency structure on the model and guide:
    variables outside of an :class:`~pyro.iarange` can never depend on
    variables inside that :class:`~pyro.iarange`.
    """

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a trace generator
        """
        guide = poutine.enum(guide, first_available_dim=self.max_iarange_nesting)
        return iter_importance_traces(num_particles=self.num_particles,
                                      graph_type="flat",
                                      max_iarange_nesting=self.max_iarange_nesting)(
                                          model, guide, *args, **kwargs)

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Estimates the ELBO using ``num_particles`` many samples (particles).
        """
        elbo = 0.0
        for _, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = _compute_dice_elbo(model_trace, guide_trace)
            if is_identically_zero(elbo_particle):
                continue

            elbo += elbo_particle.item() / self.num_particles

        loss = -elbo
        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Estimates the ELBO using ``num_particles`` many samples (particles).
        Performs backward on the ELBO of each particle.
        """
        elbo = 0.0
        for _, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = _compute_dice_elbo(model_trace, guide_trace)
            if is_identically_zero(elbo_particle):
                continue

            elbo += elbo_particle.item() / self.num_particles

            # collect parameters to train from model and guide
            trainable_params = set(site["value"].unconstrained()
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values()
                                   if site["type"] == "param")

            if trainable_params and elbo_particle.requires_grad:
                loss_particle = -elbo_particle
                (loss_particle / self.num_particles).backward()
                pyro.get_param_store().mark_params_active(trainable_params)

        loss = -elbo
        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
