from __future__ import absolute_import, division, print_function

import weakref
from collections import defaultdict

import torch

import pyro
import pyro.ops.jit
from pyro.distributions.util import is_identically_zero
from pyro.infer import ELBO
from pyro.infer.enum import get_importance_trace
from pyro.infer.util import (MultiFrameTensor, detach_iterable, get_plate_stacks,
                             is_validation_enabled, torch_backward, torch_item)
from pyro.util import check_if_enumerated, warn_if_nan


def _get_baseline_options(site):
    """
    Extracts baseline options from ``site["infer"]["baseline"]``.
    """
    # XXX default for baseline_beta currently set here
    options_dict = site["infer"].get("baseline", {}).copy()
    options_tuple = (options_dict.pop('nn_baseline', None),
                     options_dict.pop('nn_baseline_input', None),
                     options_dict.pop('use_decaying_avg_baseline', False),
                     options_dict.pop('baseline_beta', 0.90),
                     options_dict.pop('baseline_value', None))
    if options_dict:
        raise ValueError("Unrecognized baseline options: {}".format(options_dict.keys()))
    return options_tuple


def _compute_downstream_costs(model_trace, guide_trace, non_reparam_nodes):

    # Computes a surrogate cost for each nonreparam node in the guide. It would
    # be equivalent (in expectation), to use the full ELBO as a cost for each
    # node. But for many models/guides, there will be "upstream" terms in the
    # ELBO which would only add variance to the gradient estimates.
    #
    # As in [1], only downstream costs are included. This multitree structure
    # motivates a recursive computation. Note, while only the downstream costs
    # for nonreparameterized guide nodes will ultimately be used, the reparam
    # node costs count toward those downstream totals.

    downstream_costs = {}

    # While accumulating costs, track which terms have been included for each
    # node to support detecting and handling redundant paths.
    included_model_terms = defaultdict(lambda: set())
    included_guide_terms = defaultdict(lambda: set())

    def charge(paying_node, cost):
        # Add to the downstream total of paying_node
        if paying_node in downstream_costs:
            downstream_costs[paying_node].add(*cost.items())
        else:
            downstream_costs[paying_node] = cost

    stacks = get_plate_stacks(model_trace)

    def charge_individual(paying_node, cost_node, model_term=True, guide_term=True):

        log_prob = 0.0
        if model_term:
            log_prob += model_trace.nodes[cost_node]['log_prob']
            included_model_terms[paying_node].add(cost_node)
        if guide_term:
            log_prob -= guide_trace.nodes[cost_node]['log_prob']
            included_guide_terms[paying_node].add(cost_node)

        cost = MultiFrameTensor((stacks[paying_node], log_prob))
        charge(paying_node, cost)

    def attempt_charge_downstream(paying_node, cost_node):
        # If no duplication, charge paying_node for the full
        # downstream cost from cost_node. Return any terms that *may* still
        # be needed
        required_model_terms = included_model_terms[cost_node]
        required_guide_terms = included_guide_terms[cost_node]

        already_model_terms = included_model_terms[paying_node]
        already_guide_terms = included_guide_terms[paying_node]

        needs_all = (
            required_model_terms.isdisjoint(already_model_terms) and
            required_guide_terms.isdisjoint(already_guide_terms)
        )

        if needs_all:
            charge(paying_node, downstream_costs[cost_node])
            already_model_terms.update(required_model_terms)
            already_guide_terms.update(required_guide_terms)
            return set(), set()

        return required_model_terms, required_guide_terms

    # The processing proceeds from later to earlier nodes in the guide
    guide_nodes = [x for x in guide_trace.topological_sort(reverse=True)
                   if guide_trace.nodes[x]["type"] == "sample"]
    guide_node_to_index = dict((i, n) for n, i in enumerate(guide_nodes))

    # From later to earlier...
    for node in guide_nodes:
        charge_individual(node, node)
        # Process downstream nodes from earlier to later to potentially reduce
        # the number of operations
        children = sorted(
            guide_trace.successors(node),
            key=lambda n: guide_node_to_index[n],
            reverse=True
        )
        required_model_terms = set()
        required_guide_terms = set()
        for child in children:
            # In the case of duplicates, reduce the chance of further collisions
            # by deferring adding the missing terms
            more_model_terms, more_guide_terms = attempt_charge_downstream(node, child)
            required_model_terms.update(more_model_terms)
            required_guide_terms.update(more_guide_terms)

        # Add any missing terms
        for missing_node in required_model_terms.difference(included_model_terms[node]):
            charge_individual(node, missing_node, guide_term=False)
        for missing_node in required_guide_terms.difference(included_guide_terms[node]):
            charge_individual(node, missing_node, model_term=False)

    # Add any missing terms for children in the model of downstream guide nodes
    for node in non_reparam_nodes:
        possibly_missing_terms = set()
        for downstream_node in included_guide_terms[node]:
            possibly_missing_terms.update(model_trace.successors(downstream_node))
        missing_model_terms = possibly_missing_terms.difference(included_model_terms[node])
        [charge_individual(node, n, guide_term=False) for n in missing_model_terms]

    # Collapse the conditionally independent stacks for needed costs
    for node in non_reparam_nodes:
        downstream_costs[node] = downstream_costs[node].sum_to(guide_trace.nodes[node]["cond_indep_stack"])

    return downstream_costs, included_model_terms, included_guide_terms


def _compute_elbo_reparam(model_trace, guide_trace):

    # Compute a surrogate ELBO. In [1], this is simply the ELBO because
    # (non)reparameterization is captured in the structure of the computation
    # graph, and the required behavior under parameter differentiation emerges
    # from that structure. Here the behavior of E_p[log(p)] under parameter
    # differentiation is instead represented in site["score_parts"].

    elbo = 0.0
    surrogate_elbo = 0.0

    # Bring log p(x, z|...) terms into both the ELBO and the surrogate
    for name, site in model_trace.nodes.items():
        if site["type"] == "sample":
            elbo += site["log_prob_sum"]
            surrogate_elbo += site["log_prob_sum"]

    # Bring log q(z|...) terms into the ELBO, and effective terms into the
    # surrogate
    for name, site in guide_trace.nodes.items():
        if site["type"] == "sample":
            elbo -= site["log_prob_sum"]
            entropy_term = site["score_parts"].entropy_term
            # For fully reparameterized terms, this entropy_term is log q(z|...)
            # For fully non-reparameterized terms, it is zero
            if not is_identically_zero(entropy_term):
                surrogate_elbo -= entropy_term.sum()

    return elbo, surrogate_elbo


def _construct_baseline(node, guide_site, downstream_cost):

    # XXX should the average baseline be in the param store as below?

    baseline = 0.0
    baseline_loss = 0.0

    (nn_baseline, nn_baseline_input, use_decaying_avg_baseline, baseline_beta,
        baseline_value) = _get_baseline_options(guide_site)

    use_nn_baseline = nn_baseline is not None
    use_baseline_value = baseline_value is not None

    use_baseline = use_nn_baseline or use_decaying_avg_baseline or use_baseline_value

    assert(not (use_nn_baseline and use_baseline_value)), \
        "cannot use baseline_value and nn_baseline simultaneously"
    if use_decaying_avg_baseline:
        dc_shape = downstream_cost.shape
        param_name = "__baseline_avg_downstream_cost_" + node
        with torch.no_grad():
            avg_downstream_cost_old = pyro.param(param_name,
                                                 torch.zeros(dc_shape, device=guide_site['value'].device))
            avg_downstream_cost_new = (1 - baseline_beta) * downstream_cost + \
                baseline_beta * avg_downstream_cost_old
        pyro.get_param_store()[param_name] = avg_downstream_cost_new
        baseline += avg_downstream_cost_old
    if use_nn_baseline:
        # block nn_baseline_input gradients except in baseline loss
        baseline += nn_baseline(detach_iterable(nn_baseline_input))
    elif use_baseline_value:
        # it's on the user to make sure baseline_value tape only points to baseline params
        baseline += baseline_value
    if use_nn_baseline or use_baseline_value:
        # accumulate baseline loss
        baseline_loss += torch.pow(downstream_cost.detach() - baseline, 2.0).sum()

    if use_baseline:
        if downstream_cost.shape != baseline.shape:
            raise ValueError("Expected baseline at site {} to be {} instead got {}".format(
                node, downstream_cost.shape, baseline.shape))

    return use_baseline, baseline_loss, baseline


def _compute_elbo_non_reparam(guide_trace, non_reparam_nodes, downstream_costs):
    # construct all the reinforce-like terms.
    # we include only downstream costs to reduce variance
    # optionally include baselines to further reduce variance
    surrogate_elbo = 0.0
    baseline_loss = 0.0
    for node in non_reparam_nodes:
        guide_site = guide_trace.nodes[node]
        downstream_cost = downstream_costs[node]
        score_function = guide_site["score_parts"].score_function

        use_baseline, baseline_loss_term, baseline = _construct_baseline(node, guide_site, downstream_cost)

        if use_baseline:
            downstream_cost -= baseline
            baseline_loss += baseline_loss_term

        surrogate_elbo += (score_function * downstream_cost.detach()).sum()

    return surrogate_elbo, baseline_loss


class TraceGraph_ELBO(ELBO):
    """
    A TraceGraph implementation of ELBO-based SVI. The gradient estimator
    is constructed along the lines of reference [1] specialized to the case
    of the ELBO. It supports arbitrary dependency structure for the model
    and guide as well as baselines for non-reparameterizable random variables.
    Where possible, conditional dependency information as recorded in the
    :class:`~pyro.poutine.trace.Trace` is used to reduce the variance of the gradient estimator.
    In particular two kinds of conditional dependency information are
    used to reduce variance:

    - the sequential order of samples (z is sampled after y => y does not depend on z)
    - :class:`~pyro.plate` generators

    References

    [1] `Gradient Estimation Using Stochastic Computation Graphs`,
        John Schulman, Nicolas Heess, Theophane Weber, Pieter Abbeel

    [2] `Neural Variational Inference and Learning in Belief Networks`
        Andriy Mnih, Karol Gregor
    """

    def _get_trace(self, model, guide, *args, **kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "dense", self.max_plate_nesting, model, guide, *args, **kwargs)
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = torch_item(model_trace.log_prob_sum()) - torch_item(guide_trace.log_prob_sum())
            elbo += elbo_particle / float(self.num_particles)

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        If baselines are present, a baseline loss is also constructed and differentiated.
        """

        loss, surrogate_loss = self._loss_and_surrogate_loss(model, guide, *args, **kwargs)

        torch_backward(surrogate_loss)

        loss = torch_item(loss)
        warn_if_nan(loss, "loss")
        return loss

    def _loss_and_surrogate_loss(self, model, guide, *args, **kwargs):

        loss = 0.0
        surrogate_loss = 0.0

        for model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):

            lp, slp = self._loss_and_surrogate_loss_particle(model_trace, guide_trace, *args, **kwargs)
            loss += lp
            surrogate_loss += slp

        loss /= self.num_particles
        surrogate_loss /= self.num_particles

        return loss, surrogate_loss

    def _loss_and_surrogate_loss_particle(self, model_trace, guide_trace, *args, **kwargs):

        # compute elbo for reparameterized nodes
        elbo, surrogate_elbo = _compute_elbo_reparam(model_trace, guide_trace)
        baseline_loss = 0.0

        # the following computations are only necessary if we have non-reparameterizable nodes
        non_reparam_nodes = set(guide_trace.nonreparam_stochastic_nodes)
        if non_reparam_nodes:
            downstream_costs, _, _ = _compute_downstream_costs(model_trace, guide_trace, non_reparam_nodes)
            surrogate_elbo_term, baseline_loss = _compute_elbo_non_reparam(guide_trace,
                                                                           non_reparam_nodes,
                                                                           downstream_costs)
            surrogate_elbo += surrogate_elbo_term

        surrogate_loss = -surrogate_elbo + baseline_loss

        return elbo, surrogate_loss


class JitTraceGraph_ELBO(TraceGraph_ELBO):
    """
    Like :class:`TraceGraph_ELBO` but uses :func:`torch.jit.trace` to
    compile :meth:`loss_and_grads`.

    This works only for a limited set of models:

    -   Models must have static structure.
    -   Models must not depend on any global data (except the param store).
    -   All model inputs that are tensors must be passed in via ``*args``.
    -   All model inputs that are *not* tensors must be passed in via
        ``**kwargs``, and compilation will be triggered once per unique
        ``**kwargs``.
    """

    def loss_and_grads(self, model, guide, *args, **kwargs):
        kwargs['_pyro_model_id'] = id(model)
        kwargs['_pyro_guide_id'] = id(guide)
        if getattr(self, '_jit_lsl', None) is None:
            # build a closure for loss_and_surrogate_loss
            weakself = weakref.ref(self)

            @pyro.ops.jit.trace(ignore_warnings=self.ignore_jit_warnings,
                                jit_options=self.jit_options)
            def jit_lsl(*args, **kwargs):
                kwargs.pop('_pyro_model_id')
                kwargs.pop('_pyro_guide_id')
                self = weakself()
                return self._loss_and_surrogate_loss(model, guide, *args, **kwargs)

            self._jit_lsl = jit_lsl

        loss, surrogate_loss = self._jit_lsl(*args, **kwargs)

        surrogate_loss.backward()  # triggers jit compilation

        loss = loss.item()
        warn_if_nan(loss, "loss")
        return loss
