from collections import OrderedDict, defaultdict
from functools import partial

import torch
from opt_einsum import shared_intermediates

from pyro.distributions.util import broadcast_shape, logsumexp
from pyro.infer.util import is_validation_enabled
from pyro.ops.contract import contract_to_tensor
from pyro.poutine.subsample_messenger import _Subsample
from pyro.util import check_site_shape


class TraceTreeEvaluator(object):
    """
    Computes the log probability density of a trace (of a model with
    tree structure) that possibly contains discrete sample sites
    enumerated in parallel. This will be deprecated in favor of
    :class:`~pyro.infer.mcmc.util.EinsumTraceProbEvaluator`.

    :param model_trace: execution trace from a static model.
    :param bool has_enumerable_sites: whether the trace contains any
        discrete enumerable sites.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts.
    """
    def __init__(self,
                 model_trace,
                 has_enumerable_sites=False,
                 max_plate_nesting=None):
        self.has_enumerable_sites = has_enumerable_sites
        self.max_plate_nesting = max_plate_nesting
        # To be populated using the model trace once.
        self._log_probs = defaultdict(list)
        self._log_prob_shapes = defaultdict(tuple)
        self._children = defaultdict(list)
        self._enum_dims = {}
        self._plate_dims = {}
        self._parse_model_structure(model_trace)

    def _parse_model_structure(self, model_trace):
        if not self.has_enumerable_sites:
            return
        if self.max_plate_nesting is None:
            raise ValueError("Finite value required for `max_plate_nesting` when model "
                             "has discrete (enumerable) sites.")
        self._compute_log_prob_terms(model_trace)
        # 1. Infer model structure - compute parent-child relationship.
        sorted_ordinals = sorted(self._log_probs.keys())
        for i, child_node in enumerate(sorted_ordinals):
            for j in range(i-1, -1, -1):
                cur_node = sorted_ordinals[j]
                if cur_node < child_node:
                    self._children[cur_node].append(child_node)
                    break  # at most 1 parent.
        # 2. Populate `plate_dims` and `enum_dims` to be evaluated/
        #    enumerated out at each ordinal.
        self._populate_cache(frozenset(), frozenset(), set())

    def _populate_cache(self, ordinal, parent_ordinal, parent_enum_dims):
        """
        For each ordinal, populate the `plate` and `enum` dims to be
        evaluated or enumerated out.
        """
        log_prob_shape = self._log_prob_shapes[ordinal]
        plate_dims = sorted([frame.dim for frame in ordinal - parent_ordinal])
        enum_dims = set((i for i in range(-len(log_prob_shape), -self.max_plate_nesting)
                         if log_prob_shape[i] > 1))
        self._plate_dims[ordinal] = plate_dims
        self._enum_dims[ordinal] = set(enum_dims - parent_enum_dims)
        for c in self._children[ordinal]:
            self._populate_cache(c, ordinal, enum_dims)

    def _compute_log_prob_terms(self, model_trace):
        """
        Computes the conditional probabilities for each of the sites
        in the model trace, and stores the result in `self._log_probs`.
        """
        model_trace.compute_log_prob()
        self._log_probs = defaultdict(list)
        ordering = {name: frozenset(site["cond_indep_stack"])
                    for name, site in model_trace.nodes.items()
                    if site["type"] == "sample"}
        # Collect log prob terms per independence context.
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                if is_validation_enabled():
                    check_site_shape(site, self.max_plate_nesting)
                self._log_probs[ordering[name]].append(site["log_prob"])
        if not self._log_prob_shapes:
            for ordinal, log_prob in self._log_probs.items():
                self._log_prob_shapes[ordinal] = broadcast_shape(*(t.shape for t in self._log_probs[ordinal]))

    def _reduce(self, ordinal, agg_log_prob=torch.tensor(0.)):
        """
        Reduce the log prob terms for the given ordinal:
          - taking log_sum_exp of factors in enum dims (i.e.
            adding up the probability terms).
          - summing up the dims within `max_plate_nesting`.
            (i.e. multiplying probs within independent batches).

        :param ordinal: node (ordinal)
        :param torch.Tensor agg_log_prob: aggregated `log_prob`
            terms from the downstream nodes.
        :return: `log_prob` with marginalized `plate` and `enum`
            dims.
        """
        log_prob = sum(self._log_probs[ordinal]) + agg_log_prob
        for enum_dim in self._enum_dims[ordinal]:
            log_prob = logsumexp(log_prob, dim=enum_dim, keepdim=True)
        for marginal_dim in self._plate_dims[ordinal]:
            log_prob = log_prob.sum(dim=marginal_dim, keepdim=True)
        return log_prob

    def _aggregate_log_probs(self, ordinal):
        """
        Aggregate the `log_prob` terms using depth first search.
        """
        if not self._children[ordinal]:
            return self._reduce(ordinal)
        agg_log_prob = sum(map(self._aggregate_log_probs, self._children[ordinal]))
        return self._reduce(ordinal, agg_log_prob)

    def log_prob(self, model_trace):
        """
        Returns the log pdf of `model_trace` by appropriately handling
        enumerated log prob factors.

        :return: log pdf of the trace.
        """
        with shared_intermediates():
            if not self.has_enumerable_sites:
                return model_trace.log_prob_sum()
            self._compute_log_prob_terms(model_trace)
            return self._aggregate_log_probs(ordinal=frozenset()).sum()


class TraceEinsumEvaluator(object):
    """
    Computes the log probability density of a trace (of a model with
    tree structure) that possibly contains discrete sample sites
    enumerated in parallel. This uses optimized `einsum` operations
    to marginalize out the the enumerated dimensions in the trace
    via :class:`~pyro.ops.contract.contract_to_tensor`.

    :param model_trace: execution trace from a static model.
    :param bool has_enumerable_sites: whether the trace contains any
        discrete enumerable sites.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts.
    """
    def __init__(self,
                 model_trace,
                 has_enumerable_sites=False,
                 max_plate_nesting=None):
        self.has_enumerable_sites = has_enumerable_sites
        self.max_plate_nesting = max_plate_nesting
        # To be populated using the model trace once.
        self._enum_dims = set()
        self.ordering = {}
        self._populate_cache(model_trace)

    def _populate_cache(self, model_trace):
        """
        Populate the ordinals (set of ``CondIndepStack`` frames)
        and enum_dims for each sample site.
        """
        if not self.has_enumerable_sites:
            return
        if self.max_plate_nesting is None:
            raise ValueError("Finite value required for `max_plate_nesting` when model "
                             "has discrete (enumerable) sites.")
        model_trace.compute_log_prob()
        model_trace.pack_tensors()
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and not isinstance(site["fn"], _Subsample):
                if is_validation_enabled():
                    check_site_shape(site, self.max_plate_nesting)
                self.ordering[name] = frozenset(model_trace.plate_to_symbol[f.name]
                                                for f in site["cond_indep_stack"]
                                                if f.vectorized)
        self._enum_dims = set(model_trace.symbol_to_dim) - set(model_trace.plate_to_symbol.values())

    def _get_log_factors(self, model_trace):
        """
        Aggregates the `log_prob` terms into a list for each
        ordinal.
        """
        model_trace.compute_log_prob()
        model_trace.pack_tensors()
        log_probs = OrderedDict()
        # Collect log prob terms per independence context.
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample" and not isinstance(site["fn"], _Subsample):
                if is_validation_enabled():
                    check_site_shape(site, self.max_plate_nesting)
                log_probs.setdefault(self.ordering[name], []).append(site["packed"]["log_prob"])
        return log_probs

    def log_prob(self, model_trace):
        """
        Returns the log pdf of `model_trace` by appropriately handling
        enumerated log prob factors.

        :return: log pdf of the trace.
        """
        if not self.has_enumerable_sites:
            return model_trace.log_prob_sum()
        log_probs = self._get_log_factors(model_trace)
        with shared_intermediates() as cache:
            return contract_to_tensor(log_probs, self._enum_dims, cache=cache)


def _guess_max_plate_nesting(model, args, kwargs):
    """
    Guesses max_plate_nesting by running the model once
    without enumeration. This optimistically assumes static model
    structure.
    """
    with poutine.block():
        model_trace = poutine.trace(model).get_trace(*args, **kwargs)
    sites = [site for site in model_trace.nodes.values()
             if site["type"] == "sample"]

    dims = [frame.dim
            for site in sites
            for frame in site["cond_indep_stack"]
            if frame.vectorized]
    max_plate_nesting = -min(dims) if dims else 0
    return max_plate_nesting


def _pe_maker(model, model_args, model_kwargs, transforms, trace_prob_evaluator):
    def potential_energy(params):
        params_constrained = {k: transforms[k].inv(v) for k, v in params.items()}
        model = poutine.condition(model, params_constrained)
        model_trace = poutine.trace(model).get_trace(*model_args, **model_kwargs)
        log_joint = -trace_prob_evaluator.log_prob(model_trace)
        for name, t in transforms.items():
            log_joint = log_joint + torch.sum(
                t.log_abs_det_jacobian(params_constrained[name], params[name]))
        return log_joint

    return potential_energy


def _transform_fn(transforms, params, invert=False):
    return {k: transforms[k](v) if not invert else transforms[k].inv(v)
            for k, v in params.items()}


def initial_params():
    max_tries_initial_trace = 100


def initialize_model(model, model_args, model_kwargs, transforms=None, max_plate_nesting=None,
                     jit_compile=False, jit_options=None, ignore_jit_warnings=False):
    # XXX `transforms` domains are sites' supports
    if transforms is None:
        automatic_transform_enabled = True
        transforms = {}
    else:
        automatic_transform_enabled = False
    max_plate_nesting = _guess_max_plate_nesting(model, model_args, model_kwargs)
    # Wrap model in `poutine.enum` to enumerate over discrete latent sites.
    # No-op if model does not have any discrete latents.
    model = poutine.enum(config_enumerate(model),
                         first_available_dim=-1 - max_plate_nesting)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    has_enumerable_sites = False
    sample_sites = {}
    for name, node in trace.iter_stochastic_nodes():
        if isinstance(node["fn"], _Subsample):
            continue
        if node["fn"].has_enumerate_support:
            has_enumerable_sites = True
            continue
        sample_sites[name] = node["value"]
        if node["fn"].support is not constraints.real and automatic_transform_enabled:
            transforms[name] = biject_to(node["fn"].support).inv
        site_value = transforms[name](node["value"])

    trace_prob_evaluator = TraceEinsumEvaluator(model_trace,
                                                has_enumerable_sites,
                                                max_plate_nesting)
    # TODO: max tries initial_params
    init_params = _transform_fn(transforms,
                                {k: v['value'] for k, v in sample_sites.items()},
                                invert=True)
    potential_fn = _pe_maker(model, model_args, model_kwargs, transforms)

    if jit_compile:
        jit_options = {"check_trace": False} if jit_options is None else jit_options
        with pyro.validation_enabled(False), optional(ignore_jit_warnings(), ignore_jit_warnings):
            names, vals = zip(*sorted(sample_sites.items()))

            def _pe_jit(*zi):
                params = dict(zip(names, zi))
                return potential_fn(params)

            compiled_pe = torch.jit.trace(_pe_jit, vals, **jit_options)

            def potental_fn(params):
                _, vals = zip(*sorted(params.items()))
                return compiled_pe(*vals)

    return init_params, potential_fn, partial(transform_fn, transforms)
