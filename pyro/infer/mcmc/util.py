from collections import defaultdict

from pyro.distributions.util import logsumexp
from pyro.infer.util import is_validation_enabled
from pyro.util import check_site_shape


class EnumTraceProbEvaluator(object):
    """
    Computes the log probability density of a trace that possibly contains
    discrete sample sites enumerated in parallel.

    :param model_trace: execution trace from a static model.
    :param bool has_enumerable_sites: whether the trace contains any
        discrete enumerable sites.
    :param int max_iarange_nesting: Optional bound on max number of nested
        :func:`pyro.iarange` contexts.
    """
    def __init__(self,
                 model_trace,
                 has_enumerable_sites=False,
                 max_iarange_nesting=float("inf")):
        self.has_enumerable_sites = has_enumerable_sites
        self.max_iarange_nesting = max_iarange_nesting
        # To be populated using the model trace once.
        self._log_probs = {}
        self._children = defaultdict(list)
        self._enum_dims = {}
        self._iarange_dims = {}
        self._parse_model_structure(model_trace)

    def _parse_model_structure(self, model_trace):
        if not self.has_enumerable_sites:
            return
        if self.max_iarange_nesting == float("inf"):
            raise ValueError("Finite value required for `max_iarange_nesting` when model "
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
        # 2. Populate `iarange_dims` and `enum_dims` to be evaluated/
        #    enumerated out at each ordinal.
        self._populate_cache(frozenset(), frozenset(), set())

    def _populate_cache(self, ordinal, parent_ordinal, parent_enum_dims):
        """
        For each ordinal, populate the `iarange` and `enum` dims to be
        evaluated or enumerated out.
        """
        log_prob = self._log_probs[ordinal]
        iarange_dims = sorted([frame.dim for frame in ordinal - parent_ordinal])
        enum_dims = set((i for i in range(-log_prob.dim(), -self.max_iarange_nesting)
                         if log_prob.shape[i] > 1))
        self._iarange_dims[ordinal] = iarange_dims
        self._enum_dims[ordinal] = sorted(enum_dims - parent_enum_dims)
        for c in self._children[ordinal]:
            self._populate_cache(c, ordinal, enum_dims)

    def _compute_log_prob_terms(self, model_trace):
        """
        Computes the conditional probabilities for each of the sites
        in the model trace, and stores the result in `self._log_probs`.
        """
        model_trace.compute_log_prob()
        ordering = {name: frozenset(site["cond_indep_stack"])
                    for name, site in model_trace.nodes.items()
                    if site["type"] == "sample"}
        # Collect log prob terms per independence context.
        log_probs = defaultdict(list)
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                if is_validation_enabled():
                    check_site_shape(site, self.max_iarange_nesting)
                log_probs[ordering[name]].append(site["log_prob"])

        for ordinal, log_prob in log_probs.items():
            self._log_probs[ordinal] = sum(log_prob)

    def _reduce(self, ordinal, agg_log_prob=0.):
        """
        Reduce the log prob terms for the given ordinal:
          - taking log_sum_exp of factors in enum dims (i.e.
            adding up the probability terms).
          - summing up the dims within `max_iarange_nesting`.
            (i.e. multiplying probs within independent batches).

        :param ordinal: node (ordinal)
        :param torch.Tensor agg_log_prob: aggregated `log_prob`
            terms from the downstream nodes.
        :return: `log_prob` with marginalized `iarange` and `enum`
            dims.
        """
        log_prob = self._log_probs[ordinal] + agg_log_prob
        for enum_dim in self._enum_dims[ordinal]:
            log_prob = logsumexp(log_prob, dim=enum_dim, keepdim=True)
        for marginal_dim in self._iarange_dims[ordinal]:
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
        if not self.has_enumerable_sites:
            return model_trace.log_prob_sum()
        self._compute_log_prob_terms(model_trace)
        return self._aggregate_log_probs(ordinal=frozenset()).sum()
