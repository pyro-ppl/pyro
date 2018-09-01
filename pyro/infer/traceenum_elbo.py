from __future__ import absolute_import, division, print_function

import warnings
import weakref
from collections import OrderedDict, defaultdict

import torch
from six.moves import queue

import pyro
import pyro.ops.jit
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape, is_identically_zero, scale_and_mask
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace, iter_discrete_escape, iter_discrete_extend
from pyro.infer.util import Dice, is_validation_enabled
from pyro.ops.sumproduct import logsumproductexp
from pyro.poutine.enumerate_messenger import EnumerateMessenger
from pyro.util import check_traceenum_requirements, warn_if_nan


def _check_model_enumeration_requirements(log_factors, scales):
    # Check for absence of diamonds, i.e. collider iaranges.
    # This is required to avoid loops in message passing.
    for t in log_factors:
        for u in log_factors:
            if not (u < t):
                continue
            for v in log_factors:
                if not (v < t):
                    continue
                if u <= v or v <= u:
                    continue
                left = ', '.join(sorted(f.name for f in u - v))
                right = ', '.join(sorted(f.name for f in v - u))
                raise ValueError("Expected tree-structured iarange nesting, but found "
                                 "dependencies on independent iarange sets [{}] and [{}]. "
                                 "Try converting one of the iaranges to an irange (but beware "
                                 "exponential cost in the size of that irange)"
                                 .format(left, right))

    # Check that all enumerated sites share a common subsampling scale.
    # Note that we use a cheap weak comparison by id rather than tensor value, because
    # (1) it is expensive to compare tensors by value, and (2) tensors must agree not
    # only in value but at all derivatives.
    if len(scales) != 1:
        raise ValueError("Expected all enumerated sample sites to share a common poutine.scale, "
                         "but found {} different scales.".format(len(scales)))


def _partition_terms(terms, dims):
    """
    Given a list of terms and a set of contraction dims, partitions the terms
    up into sets that must be contracted together. By separating these
    components we avoid broadcasting. This function should be deterministic.
    """
    # Construct a bipartite graph between terms and the dims in which they
    # are enumerated. This conflates terms and dims (tensors and ints).
    neighbors = OrderedDict([(t, []) for t in terms] + [(d, []) for d in dims])
    for term in terms:
        for dim in range(-term.dim(), 0):
            if dim in dims and term.shape[dim] > 1:
                neighbors[term].append(dim)
                neighbors[dim].append(term)

    # Partition the bipartite graph into connected components for contraction.
    while neighbors:
        v, pending = neighbors.popitem()
        component = OrderedDict([(v, None)])  # used as an OrderedSet
        for v in pending:
            component[v] = None
        while pending:
            v = pending.pop()
            for v in neighbors.pop(v):
                if v not in component:
                    component[v] = None
                    pending.append(v)

        # Split this connected component into tensors and dims.
        component_terms = [v for v in component if isinstance(v, torch.Tensor)]
        component_dims = set(v for v in component if not isinstance(v, torch.Tensor))
        yield component_terms, component_dims


def _contract(log_factors, enum_boundary):
    """
    Contract out all enumeration dims in a tree of log_factors in-place, via
    message passing. This function should be deterministic.

    :param OrderedDict log_factors: a dictionary mapping ordinals to lists of
        tensors. An ordinal is a frozenset of ``CondIndepStack`` frames.
    :param int enum_boundary: a dimension (counting from the right) such that
        all dimensions left of this dimension are enumeration dimensions.
    """
    # First close the set of ordinals under intersection (greatest lower bound),
    # ensuring that the ordinals are arranged in a tree structure.
    pending = list(log_factors)
    while pending:
        t = pending.pop()
        for u in list(log_factors):
            tu = t & u
            if tu not in log_factors:
                log_factors[tu] = []
                pending.append(tu)

    # Collect all enumeration dimensions.
    enum_dims = defaultdict(set)
    for t, terms in log_factors.items():
        enum_dims[t] |= set(i for term in terms
                            for i in range(-term.dim(), enum_boundary)
                            if term.shape[i] > 1)

    # Recursively combine terms in different iarange contexts.
    while len(log_factors) > 1 or any(enum_dims.values()):
        leaf = max(log_factors, key=len)
        leaf_terms = log_factors.pop(leaf)
        leaf_dims = enum_dims.pop(leaf)
        remaining_dims = set.union(set(), *enum_dims.values())
        contract_dims = leaf_dims - remaining_dims
        contract_frames = (frozenset.intersection(*(leaf - t for t in log_factors if t < leaf))
                           if log_factors else frozenset())
        parent = leaf - contract_frames
        enum_dims[parent] |= leaf_dims & remaining_dims
        log_factors.setdefault(parent, [])
        for terms, dims in _partition_terms(leaf_terms, contract_dims):

            # Eliminate any enumeration dims via a logsumproductexp() contraction.
            if dims:
                shape = list(broadcast_shape(*set(x.shape for x in terms)))
                for dim in dims:
                    shape[dim] = 1
                shape.reverse()
                while shape and shape[-1] == 1:
                    shape.pop()
                shape.reverse()
                shape = tuple(shape)
                terms = [logsumproductexp(terms, shape)]

            # Eliminate remaining iarange dims via .sum() contractions.
            for term in terms:
                for frame in contract_frames:
                    term = term.sum(frame.dim, keepdim=True)
                log_factors[parent].append(term)


# TODO move this logic into a poutine
def _compute_model_costs(model_trace, guide_trace, ordering):
    # Collect model sites that may have been enumerated in the model.
    cost_sites = OrderedDict()
    enum_sites = OrderedDict()
    enum_dims = []
    for name, site in model_trace.nodes.items():
        if site["type"] == "sample":
            if name in guide_trace or site["infer"].get("_enumerate_dim") is None:
                cost_sites.setdefault(ordering[name], []).append(site)
            else:
                enum_sites.setdefault(ordering[name], []).append(site)
                enum_dims.append(site["fn"].event_dim - site["value"].dim())
    if not enum_sites:
        return OrderedDict((t, [site["log_prob"] for site in sites_t])
                           for t, sites_t in cost_sites.items())

    # Marginalize out all variables that have been enumerated in the model.
    enum_boundary = max(enum_dims) + 1
    assert enum_boundary <= 0
    marginal_costs = OrderedDict()
    log_factors = OrderedDict()
    scales = set()
    for t, sites_t in cost_sites.items():
        for site in sites_t:
            if site["log_prob"].dim() <= -enum_boundary:
                # For sites that do not depend on an enumerated variable, proceed as usual.
                marginal_costs.setdefault(t, []).append(site["log_prob"])
            else:
                # For sites that depend on an enumerated variable, we need to apply
                # the mask inside- and the scale outside- of the log expectation.
                cost = scale_and_mask(site["unscaled_log_prob"], mask=site["mask"])
                log_factors.setdefault(t, []).append(cost)
                scales.add(site["scale"])
    if log_factors:
        for t, sites_t in enum_sites.items():
            # TODO refine this coarse dependency ordering using time and tensor shapes.
            if any(t <= u for u in log_factors):
                for site in sites_t:
                    logprob = site["unscaled_log_prob"]
                    log_factors.setdefault(t, []).append(logprob)
                    scales.add(site["scale"])
        _check_model_enumeration_requirements(log_factors, scales)
        scale = scales.pop()
        _contract(log_factors, enum_boundary)
        for t, log_factors_t in log_factors.items():
            marginal_costs_t = marginal_costs.setdefault(t, [])
            for term in log_factors_t:
                term = scale_and_mask(term, scale=scale)
                marginal_costs_t.append(term)
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
    - local parallel sampling over any sample site.

    To enumerate over a sample site in the ``guide``, mark the site with either
    ``infer={'enumerate': 'sequential'}`` or
    ``infer={'enumerate': 'parallel'}``. To configure all guide sites at once,
    use :func:`~pyro.infer.enum.config_enumerate`. To enumerate over a sample
    site in the ``model``, mark the site ``infer={'enumerate': 'parallel'}``
    and ensure the site does not appear in the ``guide``.

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
        # The model allocates enumeration dimensions after (to the left of) the guide,
        # accomplished by letting the model_enum lazily query the guide_enum for its
        # final .next_available_dim. The laziness is accomplished via a lambda.
        # Note this relies on the guide being run before the model.
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
