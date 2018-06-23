from __future__ import absolute_import, division, print_function

import warnings
import weakref

import torch
from torch.distributions.utils import broadcast_all

import pyro
import pyro.ops.jit
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import iter_discrete_traces
from pyro.infer.util import Dice, is_validation_enabled
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, check_traceenum_requirements, torch_isnan


def _dict_iadd(dict_, key, value):
    if key in dict_:
        dict_[key] = dict_[key] + value
    else:
        dict_[key] = value


def _compute_dice_elbo(model_trace, guide_trace):
    # y depends on x iff ordering[x] <= ordering[y]
    # TODO refine this coarse dependency ordering.
    ordering = {name: frozenset(f for f in site["cond_indep_stack"] if f.vectorized)
                for trace in (model_trace, guide_trace)
                for name, site in trace.nodes.items()
                if site["type"] == "sample"}

    costs = {}
    for name, site in model_trace.nodes.items():
        if site["type"] == "sample":
            _dict_iadd(costs, ordering[name], site["log_prob"])
    for name, site in guide_trace.nodes.items():
        if site["type"] == "sample":
            _dict_iadd(costs, ordering[name], -site["log_prob"])

    dice = Dice(guide_trace, ordering)
    elbo = 0.0
    for ordinal, cost in costs.items():
        dice_prob = dice.in_context(cost.shape, ordinal)
        mask = dice_prob > 0
        if torch.is_tensor(mask) and not mask.all():
            cost, dice_prob, mask = broadcast_all(cost, dice_prob, mask)
            dice_prob = dice_prob[mask]
            cost = cost[mask]
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

    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        for guide_trace in iter_discrete_traces("flat", guide, *args, **kwargs):
            model_trace = poutine.trace(poutine.replay(model, trace=guide_trace),
                                        graph_type="flat").get_trace(*args, **kwargs)

            if is_validation_enabled():
                check_model_guide_match(model_trace, guide_trace, self.max_iarange_nesting)
            guide_trace = prune_subsample_sites(guide_trace)
            model_trace = prune_subsample_sites(model_trace)
            if is_validation_enabled():
                check_traceenum_requirements(model_trace, guide_trace)

            model_trace.compute_log_prob()
            guide_trace.compute_score_parts()
            if is_validation_enabled():
                for site in model_trace.nodes.values():
                    if site["type"] == "sample":
                        check_site_shape(site, self.max_iarange_nesting)
                any_enumerated = False
                for site in guide_trace.nodes.values():
                    if site["type"] == "sample":
                        check_site_shape(site, self.max_iarange_nesting)
                        if site["infer"].get("enumerate"):
                            any_enumerated = True
                if self.strict_enumeration_warning and not any_enumerated:
                    warnings.warn('TraceEnum_ELBO found no sample sites configured for enumeration. '
                                  'If you want to enumerate sites, you need to @config_enumerate or set '
                                  'infer={"enumerate": "sequential"} or infer={"enumerate": "parallel"}? '
                                  'If you do not want to enumerate, consider using Trace_ELBO instead.')

            yield model_trace, guide_trace

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        Runs the guide and runs the model against the guide with
        the result packaged as a trace generator.
        """
        if self.vectorize_particles:
            # enable parallel enumeration over the vectorized guide.
            guide = poutine.enum(self._vectorized_num_particles(guide),
                                 first_available_dim=self.max_iarange_nesting)
            model = self._vectorized_num_particles(model)
            for model_trace, guide_trace in self._get_trace(model, guide, *args, **kwargs):
                yield model_trace, guide_trace
        else:
            # enable parallel enumeration.
            guide = poutine.enum(guide, first_available_dim=self.max_iarange_nesting)
            for i in range(self.num_particles):
                for model_trace, guide_trace in self._get_trace(model, guide, *args, **kwargs):
                    yield model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Estimates the ELBO using ``num_particles`` many samples (particles).
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
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
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = _compute_dice_elbo(model_trace, guide_trace)
            if is_identically_zero(elbo_particle):
                continue

            elbo += elbo_particle.item() / self.num_particles

            # collect parameters to train from model and guide
            trainable_params = any(site["type"] == "param"
                                   for trace in (model_trace, guide_trace)
                                   for site in trace.nodes.values())

            if trainable_params and elbo_particle.requires_grad:
                loss_particle = -elbo_particle
                (loss_particle / self.num_particles).backward(retain_graph=True)

        loss = -elbo
        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss


class JitTraceEnum_ELBO(TraceEnum_ELBO):
    """
    Like :class:`TraceEnum_ELBO` but uses :func:`pyro.ops.jit.compile` to
    compile :meth:`loss_and_grads`.

    This works only for a limited set of models:

    -   Models must have static structure.
    -   Models must not depend on any global data (except the param store).
    -   All model inputs that are tensors must be passed in via ``*args``.
    -   All model inputs that are *not* tensors must be passed in via
        ``*kwargs``, and these will be fixed to their values on the first
        call to :meth:`jit_loss_and_grads`.

    .. warning:: Experimental. Interface subject to change.
    """
    def loss_and_grads(self, model, guide, *args, **kwargs):
        if getattr(self, '_differentiable_loss', None) is None:

            weakself = weakref.ref(self)

            @pyro.ops.jit.compile(nderivs=1)
            def differentiable_loss(*args):
                self = weakself()
                elbo = 0.0
                for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
                    elbo += _compute_dice_elbo(model_trace, guide_trace)
                return elbo * (-1.0 / self.num_particles)

            self._differentiable_loss = differentiable_loss

        differentiable_loss = self._differentiable_loss(*args)
        differentiable_loss.backward()  # this line triggers jit compilation
        loss = differentiable_loss.item()

        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
