from __future__ import absolute_import, division, print_function

import warnings
import weakref
from collections import OrderedDict

import torch
from opt_einsum import shared_intermediates
from six.moves import queue
from torch.distributions.utils import broadcast_all

import pyro
import pyro.distributions as dist
import pyro.ops.jit
import pyro.poutine as poutine
from pyro.distributions.torch_distribution import ReshapedDistribution
from pyro.distributions.util import is_identically_zero, scale_and_mask
from pyro.infer.contract import contract_tensor_tree, contract_to_tensor
from pyro.infer.elbo import ELBO
from pyro.infer.enum import get_importance_trace, iter_discrete_escape, iter_discrete_extend
from pyro.infer.util import Dice, is_validation_enabled
from pyro.poutine.enumerate_messenger import EnumerateMessenger
from pyro.util import check_traceenum_requirements, warn_if_nan


def _check_shared_scale(scales):
    # Check that all enumerated sites share a common subsampling scale.
    # Note that we use a cheap weak comparison by id rather than tensor value, because
    # (1) it is expensive to compare tensors by value, and (2) tensors must agree not
    # only in value but at all derivatives.
    if len(scales) != 1:
        raise ValueError("Expected all enumerated sample sites to share a common poutine.scale, "
                         "but found {} different scales.".format(len(scales)))


def _check_model_guide_enumeration_constraint(model_enum_sites, guide_trace):
    min_ordinal = frozenset.intersection(*model_enum_sites.keys())
    for name, site in guide_trace.nodes.items():
        if site["type"] == "sample" and site["infer"].get("_enumerate_dim") is not None:
            for f in site["cond_indep_stack"]:
                if f.vectorized and f not in min_ordinal:
                    raise ValueError("Expected model enumeration to be no more global than guide enumeration, "
                                     "but found model enumeration sites upstream of guide site '{}' in iarange('{}'). "
                                     "Try converting some model enumeration sites to guide enumeration sites."
                                     .format(name, f.name))


# TODO move this logic into a poutine
def _compute_model_factors(model_trace, guide_trace):
    # y depends on x iff ordering[x] <= ordering[y]
    # TODO refine this coarse dependency ordering using time.
    ordering = {name: frozenset(f for f in site["cond_indep_stack"] if f.vectorized)
                for trace in (model_trace, guide_trace)
                for name, site in trace.nodes.items()
                if site["type"] == "sample"}

    # Collect model sites that may have been enumerated in the model.
    cost_sites = OrderedDict()
    enum_sites = OrderedDict()
    enum_dims = []
    for name, site in model_trace.nodes.items():
        if site["type"] == "sample":
            if name in guide_trace.nodes or site["infer"].get("_enumerate_dim") is None:
                cost_sites.setdefault(ordering[name], []).append(site)
            else:
                enum_sites.setdefault(ordering[name], []).append(site)
                enum_dims.append(site["fn"].event_dim - site["value"].dim())
    log_factors = OrderedDict()
    sum_dims = {}
    scale = 1
    if not enum_sites:
        marginal_costs = OrderedDict((t, [site["log_prob"] for site in sites_t])
                                     for t, sites_t in cost_sites.items())
        return marginal_costs, log_factors, ordering, sum_dims, scale
    _check_model_guide_enumeration_constraint(enum_sites, guide_trace)

    # Marginalize out all variables that have been enumerated in the model.
    enum_boundary = max(enum_dims) + 1
    assert enum_boundary <= 0
    marginal_costs = OrderedDict()
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
        _check_shared_scale(scales)
        scale = scales.pop()
    sum_dims = {x: set(i for i in range(-x.dim(), enum_boundary) if x.shape[i] > 1)
                for xs in log_factors.values() for x in xs}
    return marginal_costs, log_factors, ordering, sum_dims, scale


def _compute_dice_elbo(model_trace, guide_trace):
    # Accumulate marginal model costs.
    marginal_costs, log_factors, ordering, sum_dims, scale = _compute_model_factors(
            model_trace, guide_trace)
    if log_factors:
        log_factors = contract_tensor_tree(log_factors, sum_dims)
        for t, log_factors_t in log_factors.items():
            marginal_costs_t = marginal_costs.setdefault(t, [])
            for term in log_factors_t:
                term = scale_and_mask(term, scale=scale)
                marginal_costs_t.append(term)
    costs = marginal_costs

    # Accumulate negative guide costs.
    for name, site in guide_trace.nodes.items():
        if site["type"] == "sample":
            costs.setdefault(ordering[name], []).append(-site["log_prob"])

    return Dice(guide_trace, ordering).compute_expectation(costs)


def _make_dist(dist_, logits):
    # Reshape for Bernoulli vs Categorical, OneHotCategorical, etc..
    if isinstance(dist_, dist.Bernoulli):
        logits = logits[..., 1] - logits[..., 0]
    elif isinstance(dist_, ReshapedDistribution):
        return _make_dist(dist_.base_dist, logits=logits)
    return type(dist_)(logits=logits)


def _compute_marginals(model_trace, guide_trace):
    args = _compute_model_factors(model_trace, guide_trace)
    marginal_costs, log_factors, ordering, sum_dims, scale = args

    marginal_dists = OrderedDict()
    with shared_intermediates():
        for name, site in model_trace.nodes.items():
            if (site["type"] != "sample" or
                    name in guide_trace.nodes or
                    site["infer"].get("_enumerate_dim") is None):
                continue

            enum_dim = site["fn"].event_dim - site["value"].dim()
            site_sum_dims = {term: dims - {enum_dim} for term, dims in sum_dims.items()}
            ordinal = frozenset(f for f in site["cond_indep_stack"] if f.vectorized)
            logits = contract_to_tensor(log_factors, site_sum_dims, ordinal)
            logits = logits.unsqueeze(-1).transpose(-1, enum_dim - 1)
            while logits.shape[0] == 1:
                logits.squeeze_(0)
            marginal_dists[name] = _make_dist(site["fn"], logits)
    return marginal_dists


class BackwardSampleMessenger(pyro.poutine.messenger.Messenger):
    """
    Implements forward filtering / backward sampling for sampling
    from the joint posterior distribution
    """
    def __init__(self, enum_trace, guide_trace):
        self.enum_trace = enum_trace
        args = _compute_model_factors(enum_trace, guide_trace)
        self.log_factors = args[1]
        self.sum_dims = args[3]
        self.cache = None

    def __enter__(self):
        self.cache = {}
        return super(BackwardSampleMessenger, self).__enter__()

    def __exit__(self, *args, **kwargs):
        assert not any(self.sum_dims.values())
        return super(BackwardSampleMessenger, self).__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        if msg["name"] not in self.enum_trace.nodes:
            return
        enum_dim = self.enum_trace.nodes[msg["name"]]["infer"].get("_enumerate_dim")
        if enum_dim is not None:
            msg["infer"]["_enumerate_dim"] = enum_dim
            assert enum_dim < 0, "{} {}".format(msg["name"], enum_dim)
            for value in self.sum_dims.values():
                value.discard(enum_dim)
            with shared_intermediates(self.cache):
                ordinal = frozenset(f for f in msg["cond_indep_stack"] if f.vectorized)
                logits = contract_to_tensor(self.log_factors, self.sum_dims, ordinal)
                logits = logits.unsqueeze(-1).transpose(-1, enum_dim - 1)
                while logits.shape[0] == 1:
                    logits.squeeze_(0)
            msg["fn"] = _make_dist(msg["fn"], logits)

    def _postprocess_message(self, msg):
        if msg["type"] == "sample":
            enum_dim = msg["infer"].get("_enumerate_dim")
            if enum_dim is not None:
                for t, terms in self.log_factors.items():
                    for i, term in enumerate(terms):
                        if term.dim() >= -enum_dim and term.shape[enum_dim] > 1:
                            term_, value_ = broadcast_all(term, msg["value"])
                            value_ = value_.index_select(enum_dim, value_.new_tensor([0], dtype=torch.long))
                            sampled_term = term_.gather(enum_dim, value_.long())
                            terms[i] = sampled_term
                            self.sum_dims[sampled_term] = self.sum_dims.pop(term) - {enum_dim}


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

    def compute_marginals(self, model, guide, *args, **kwargs):
        """
        Computes marginal distributions at each model-enumerated sample site.

        :returns: a dict mapping site name to marginal ``Distribution`` object
        :rtype: OrderedDict
        """
        if self.num_particles != 1:
            raise NotImplementedError("TraceEnum_ELBO.compute_marginals() is not "
                                      "compatible with multiple particles.")
        model_trace, guide_trace = next(self._get_traces(model, guide, *args, **kwargs))
        for site in guide_trace.nodes.values():
            if site["type"] == "sample":
                if "_enumerate_dim" in site["infer"] or "_enum_total" in site["infer"]:
                    raise NotImplementedError("TraceEnum_ELBO.compute_marginals() is not "
                                              "compatible with guide enumeration.")
        return _compute_marginals(model_trace, guide_trace)

    def sample_posterior(self, model, guide, *args, **kwargs):
        """
        Sample from the joint posterior distribution of all model-enumerated sites given all observations
        """
        if self.num_particles != 1:
            raise NotImplementedError("TraceEnum_ELBO.sample_posterior() is not "
                                      "compatible with multiple particles.")
        with poutine.block(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", "Found vars in model but not guide")
            model_trace, guide_trace = next(self._get_traces(model, guide, *args, **kwargs))

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                if "_enumerate_dim" in site["infer"] or "_enum_total" in site["infer"]:
                    raise NotImplementedError("TraceEnum_ELBO.sample_posterior() is not "
                                              "compatible with guide enumeration.")

        with BackwardSampleMessenger(model_trace, guide_trace):
            return poutine.replay(poutine.broadcast(model),
                                  trace=guide_trace)(*args, **kwargs)


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
