from collections import defaultdict

from pyro.distributions.util import log_sum_exp
from pyro.infer.util import is_validation_enabled
from pyro.util import check_site_shape


class EnumTraceProbEvaluator(object):
    def __init__(self,
                 model_trace,
                 has_enumerable_sites=False,
                 max_iarange_nesting=float("inf")):
        """
        Computes the log probability density of a trace that possibly contains
        discrete sample sites enumerated in parallel.

        :param model_trace: execution trace from the model.
        :param bool has_enumerable_sites: whether the trace contains any
            discrete enumerable sites.
        :param int max_iarange_nesting: Optional bound on max number of nested
            :func:`pyro.iarange` contexts.
        """
        self.model_trace = model_trace
        self.has_enumerable_sites = has_enumerable_sites
        self.max_iarange_nesting = max_iarange_nesting
        self.log_probs = defaultdict(list)
        self._log_factors_cache = {}
        self._predecessors_cache = {}

    def _get_predecessors_log_factors(self, target_ordinal):
        """
        Returns the list of predecessors for `target_ordinal `and
        their log_prob factors.
        """
        if target_ordinal in self._predecessors_cache:
            return self._predecessors_cache[target_ordinal],\
                   self._log_factors_cache[target_ordinal]
        log_factors = []
        predecessors = set()

        for ordinal, term in self.log_probs.items():
            if ordinal < target_ordinal:
                log_factors += term
                predecessors.add(ordinal)

        self._log_factors_cache[target_ordinal] = log_factors
        self._predecessors_cache[target_ordinal] = predecessors
        return predecessors, log_factors

    def _compute_log_prob_terms(self):
        """
        Computes the conditional probabilities for each of the sites
        in the model trace, and stores the result in `self.log_probs`.
        """
        if len(self.log_probs) > 0:
            return
        self.model_trace.compute_log_prob()
        ordering = {name: frozenset(site["cond_indep_stack"])
                    for name, site in self.model_trace.nodes.items()
                    if site["type"] == "sample"}

        # Collect log prob terms per independence context.
        for name, site in self.model_trace.nodes.items():
            if site["type"] == "sample":
                if is_validation_enabled():
                    check_site_shape(site, self.max_iarange_nesting)
                self.log_probs[ordering[name]].append(site["log_prob"])

    def log_prob(self):
        """
        Returns the log pdf of `model_trace` by appropriate handling
        of the enumerated log prob factors.

        :return: log pdf of the trace.
        """
        if not self.has_enumerable_sites:
            return self.model_trace.log_prob_sum()
        if self.max_iarange_nesting == float("inf"):
            raise ValueError("Finite value required for `max_iarange_nesting` when model "
                             "has discrete (enumerable) sites.")
        self._compute_log_prob_terms()

        # Sum up terms from predecessor, and gather leaf nodes.
        leaves_log_probs = {}
        for target_ordinal, log_prob in self.log_probs.items():
            leaves_log_probs[target_ordinal] = log_prob
            predecessors, log_factors = self._get_predecessors_log_factors(target_ordinal)
            leaves_log_probs[target_ordinal] = sum(leaves_log_probs[target_ordinal] + log_factors)
            for ordinal in predecessors:
                if ordinal in leaves_log_probs:
                    del leaves_log_probs[ordinal]

        # Reduce the log prob terms for each leaf node:
        # - taking log_sum_exp of factors in enum dims (i.e.
        # adding up the probability terms).
        # - summing up the dims within `max_iarange_nesting`.
        # (i.e. multiplying probs within independent batches).
        log_prob_sum = 0.
        for ordinal in leaves_log_probs:
            log_prob = leaves_log_probs[ordinal]
            enum_dim = log_prob.dim() - self.max_iarange_nesting
            if enum_dim > 0:
                log_prob = log_sum_exp(log_prob.reshape(-1, *log_prob.shape[enum_dim:]), dim=0)
            log_prob_sum += log_prob.sum()
        return log_prob_sum
