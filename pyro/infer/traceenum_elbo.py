from __future__ import absolute_import, division, print_function

import warnings
import weakref
from collections import OrderedDict

import torch
from six.moves import queue

import pyro
import pyro.ops.jit
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape, is_identically_zero
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace, iter_discrete_escape, iter_discrete_extend
from pyro.infer.util import Dice, is_validation_enabled
from pyro.ops.einsum import shared_intermediates
from pyro.ops.sumproduct import logsumproductexp
from pyro.poutine.enumerate_messenger import EnumerateMessenger
from pyro.util import check_traceenum_requirements, warn_if_nan


# TODO move this logic into a poutine
def _compute_model_costs(model_trace, guide_trace, ordering):
    # Collect model log_probs, possibly marginalizing out enumerated model variables.
    costs = OrderedDict()
    enum_logprobs = OrderedDict()
    enum_dims = []
    for name, site in model_trace.nodes.items():
        if site["type"] == "sample":
            if name in guide_trace or not site["infer"].get("_enumerate_dim") is not None:
                costs.setdefault(ordering[name], []).append(site["log_prob"])
            else:
                enum_logprobs.setdefault(ordering[name], []).append(site["log_prob"])
                enum_dims.append(len(site["fn"].event_shape) - len(site["value"].shape))
    if not enum_logprobs:
        return costs

    # Marginalize out all enumerated variables.
    enum_boundary = max(enum_dims) + 1
    marginal_costs = OrderedDict((t, []) for t in costs)
    with shared_intermediates():
        for t, costs_t in costs.items():
            # TODO refine this coarse dependency ordering using time and tensor shapes.
            logprobs_t = sum((logprobs_u for u, logprobs_u in enum_logprobs.items() if u <= t), [])
            if not logprobs_t:
                marginal_costs[t] = costs_t
                continue

            ordinal_shape = broadcast_shape(*(x.shape for x in logprobs_t))
            for cost_t in costs_t:
                target_shape = broadcast_shape(cost_t.shape, ordinal_shape)
                target_shape = target_shape[enum_boundary:] if enum_boundary else ()
                marginal_costs[t].append(logsumproductexp(logprobs_t + [cost_t], target_shape))
    return marginal_costs


def _compute_dice_elbo(model_trace, guide_trace):
    # y depends on x iff ordering[x] <= ordering[y]
    # TODO refine this coarse dependency ordering using time.
    ordering = {name: frozenset(f for f in site["cond_indep_stack"] if f.vectorized)
                for trace in (model_trace, guide_trace)
                for name, site in trace.nodes.items()
                if site["type"] == "sample"}

    costs = _compute_model_costs(model_trace, guide_trace, ordering)
    for name, site in guide_trace.nodes.items():
        if site["type"] == "sample":
            costs.setdefault(ordering[name], []).append(-site["log_prob"])

    return Dice(guide_trace, ordering).compute_expectation(costs)


class TraceEnum_ELBO(ELBO):
    """
    A trace implementation of ELBO-based SVI that supports
    - exhaustive enumeration over discrete sample sites, and
    - local parallel sampling over any sample sites.

    To enumerate over a sample site, the ``guide``'s sample site must specify
    either ``infer={'enumerate': 'sequential'}`` or
    ``infer={'enumerate': 'parallel'}``. To configure all sites at once, use
    :func:`~pyro.infer.enum.config_enumerate`.

    This assumes restricted dependency structure on the model and guide:
    variables outside of an :class:`~pyro.iarange` can never depend on
    variables inside that :class:`~pyro.iarange`.
    """

    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_iarange_nesting, model, guide, *args, **kwargs)

        if is_validation_enabled():
            check_traceenum_requirements(model_trace, guide_trace)

            has_enumerated_sites = any(site["infer"].get("enumerate")
                                       for trace in (guide_trace, model_trace)
                                       for name, site in trace.nodes.items()
                                       if site["type"] == "sample")

            if self.strict_enumeration_warning and not has_enumerated_sites:
                warnings.warn('TraceEnum_ELBO found no sample sites configured for enumeration. '
                              'If you want to enumerate sites, you need to @config_enumerate or set '
                              'infer={"enumerate": "sequential"} or infer={"enumerate": "parallel"}? '
                              'If you do not want to enumerate, consider using Trace_ELBO instead.')

        return model_trace, guide_trace

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        Runs the guide and runs the model against the guide with
        the result packaged as a trace generator.
        """
        if self.vectorize_particles:
            guide = self._vectorized_num_particles(guide)
            model = self._vectorized_num_particles(model)
        else:
            guide = poutine.broadcast(guide)
            model = poutine.broadcast(model)

        # Enable parallel enumeration over the vectorized guide and model.
        # The model allocates enumeration dimensions after (to the left of) the guide.
        guide_enum = EnumerateMessenger(first_available_dim=self.max_iarange_nesting)
        model_enum = EnumerateMessenger(first_available_dim=lambda: guide_enum.next_available_dim)
        guide = guide_enum(guide)
        model = model_enum(model)
        q = queue.LifoQueue()
        guide = poutine.queue(guide, q,
                              escape_fn=iter_discrete_escape,
                              extend_fn=iter_discrete_extend)
        for i in range(1 if self.vectorize_particles else self.num_particles):
            q.put(poutine.Trace())
            while not q.empty():
                yield self._get_trace(model, guide, *args, **kwargs)

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: an estimate of the ELBO
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
        warn_if_nan(loss, "loss")
        return loss

    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        :returns: a differentiable estimate of the ELBO
        :rtype: torch.Tensor
        :raises ValueError: if the ELBO is not differentiable (e.g. is
            identically zero)

        Estimates a differentiable ELBO using ``num_particles`` many samples
        (particles).  The result should be infinitely differentiable (as long
        as underlying derivatives have been implemented).
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = _compute_dice_elbo(model_trace, guide_trace)
            if is_identically_zero(elbo_particle):
                continue

            elbo = elbo + elbo_particle
        elbo = elbo / self.num_particles

        if not torch.is_tensor(elbo) or not elbo.requires_grad:
            raise ValueError('ELBO is cannot be differentiated: {}'.format(elbo))

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: an estimate of the ELBO
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
        warn_if_nan(loss, "loss")
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

        warn_if_nan(loss, "loss")
        return loss
