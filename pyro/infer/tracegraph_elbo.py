from __future__ import absolute_import, division, print_function

import warnings
import weakref
from operator import itemgetter

import networkx
import torch

import pyro
import pyro.ops.jit
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer import ELBO
from pyro.infer.util import (MultiFrameTensor, detach_iterable, get_iarange_stacks, is_validation_enabled,
                             torch_backward, torch_item)
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, torch_isnan


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


def _compute_downstream_costs(model_trace, guide_trace,  #
                              non_reparam_nodes):
    # recursively compute downstream cost nodes for all sample sites in model and guide
    # (even though ultimately just need for non-reparameterizable sample sites)
    # 1. downstream costs used for rao-blackwellization
    # 2. model observe sites (as well as terms that arise from the model and guide having different
    # dependency structures) are taken care of via 'children_in_model' below
    topo_sort_guide_nodes = list(reversed(list(networkx.topological_sort(guide_trace))))
    topo_sort_guide_nodes = [x for x in topo_sort_guide_nodes
                             if guide_trace.nodes[x]["type"] == "sample"]
    ordered_guide_nodes_dict = {n: i for i, n in enumerate(topo_sort_guide_nodes)}

    downstream_guide_cost_nodes = {}
    downstream_costs = {}
    stacks = get_iarange_stacks(model_trace)

    for node in topo_sort_guide_nodes:
        downstream_costs[node] = MultiFrameTensor((stacks[node],
                                                   model_trace.nodes[node]['log_prob'] -
                                                   guide_trace.nodes[node]['log_prob']))
        nodes_included_in_sum = set([node])
        downstream_guide_cost_nodes[node] = set([node])
        # make more efficient by ordering children appropriately (higher children first)
        children = [(k, -ordered_guide_nodes_dict[k]) for k in guide_trace.successors(node)]
        sorted_children = sorted(children, key=itemgetter(1))
        for child, _ in sorted_children:
            child_cost_nodes = downstream_guide_cost_nodes[child]
            downstream_guide_cost_nodes[node].update(child_cost_nodes)
            if nodes_included_in_sum.isdisjoint(child_cost_nodes):  # avoid duplicates
                downstream_costs[node].add(*downstream_costs[child].items())
                # XXX nodes_included_in_sum logic could be more fine-grained, possibly leading
                # to speed-ups in case there are many duplicates
                nodes_included_in_sum.update(child_cost_nodes)
        missing_downstream_costs = downstream_guide_cost_nodes[node] - nodes_included_in_sum
        # include terms we missed because we had to avoid duplicates
        for missing_node in missing_downstream_costs:
            downstream_costs[node].add((stacks[missing_node],
                                        model_trace.nodes[missing_node]['log_prob'] -
                                        guide_trace.nodes[missing_node]['log_prob']))

    # finish assembling complete downstream costs
    # (the above computation may be missing terms from model)
    for site in non_reparam_nodes:
        children_in_model = set()
        for node in downstream_guide_cost_nodes[site]:
            children_in_model.update(model_trace.successors(node))
        # remove terms accounted for above
        children_in_model.difference_update(downstream_guide_cost_nodes[site])
        for child in children_in_model:
            assert (model_trace.nodes[child]["type"] == "sample")
            downstream_costs[site].add((stacks[child],
                                        model_trace.nodes[child]['log_prob']))
            downstream_guide_cost_nodes[site].update([child])

    for k in non_reparam_nodes:
        downstream_costs[k] = downstream_costs[k].sum_to(guide_trace.nodes[k]["cond_indep_stack"])

    return downstream_costs, downstream_guide_cost_nodes


def _compute_elbo_reparam(model_trace, guide_trace, non_reparam_nodes):
    elbo = 0.0
    surrogate_elbo = 0.0

    # deal with log p(z|...) terms
    for name, site in model_trace.nodes.items():
        if site["type"] == "sample":
            elbo += site["log_prob_sum"]
            surrogate_elbo += site["log_prob_sum"]

    # deal with log q(z|...) terms
    for name, site in guide_trace.nodes.items():
        if site["type"] == "sample":
            elbo -= site["log_prob_sum"]
            entropy_term = site["score_parts"].entropy_term
            if not is_identically_zero(entropy_term):
                surrogate_elbo -= entropy_term.sum()

    return elbo, surrogate_elbo


def _compute_elbo_non_reparam(guide_trace, non_reparam_nodes, downstream_costs):
    # construct all the reinforce-like terms.
    # we include only downstream costs to reduce variance
    # optionally include baselines to further reduce variance
    # XXX should the average baseline be in the param store as below?
    surrogate_elbo = 0.0
    baseline_loss = 0.0
    for node in non_reparam_nodes:
        guide_site = guide_trace.nodes[node]
        downstream_cost = downstream_costs[node]
        baseline = 0.0
        (nn_baseline, nn_baseline_input, use_decaying_avg_baseline, baseline_beta,
            baseline_value) = _get_baseline_options(guide_site)
        use_nn_baseline = nn_baseline is not None
        use_baseline_value = baseline_value is not None
        assert(not (use_nn_baseline and use_baseline_value)), \
            "cannot use baseline_value and nn_baseline simultaneously"
        if use_decaying_avg_baseline:
            dc_shape = downstream_cost.shape
            param_name = "__baseline_avg_downstream_cost_" + node
            with torch.no_grad():
                avg_downstream_cost_old = pyro.param(param_name,
                                                     guide_site['value'].new_zeros(dc_shape))
                avg_downstream_cost_new = (1 - baseline_beta) * downstream_cost + \
                    baseline_beta * avg_downstream_cost_old
            pyro.get_param_store().replace_param(param_name, avg_downstream_cost_new,
                                                 avg_downstream_cost_old)
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

        score_function_term = guide_site["score_parts"].score_function
        if use_nn_baseline or use_decaying_avg_baseline or use_baseline_value:
            if downstream_cost.shape != baseline.shape:
                raise ValueError("Expected baseline at site {} to be {} instead got {}".format(
                    node, downstream_cost.shape, baseline.shape))
            downstream_cost = downstream_cost - baseline
        surrogate_elbo += (score_function_term * downstream_cost.detach()).sum()

    return surrogate_elbo, baseline_loss


class TraceGraph_ELBO(ELBO):
    """
    A TraceGraph implementation of ELBO-based SVI. The gradient estimator
    is constructed along the lines of reference [1] specialized to the case
    of the ELBO. It supports arbitrary dependency structure for the model
    and guide as well as baselines for non-reparameterizable random variables.
    Where possible, conditional dependency information as recorded in the
    :class:`~pyro.poutine.trace.Trace` is used to reduce the variance of the gradient estimator.
    In particular three kinds of conditional dependency information are
    used to reduce variance:
    - the sequential order of samples (z is sampled after y => y does not depend on z)
    - :class:`~pyro.iarange` generators
    - :class:`~pyro.irange` generators

    References

    [1] `Gradient Estimation Using Stochastic Computation Graphs`,
        John Schulman, Nicolas Heess, Theophane Weber, Pieter Abbeel

    [2] `Neural Variational Inference and Learning in Belief Networks`
        Andriy Mnih, Karol Gregor
    """

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a tracegraph generator
        """

        for i in range(self.num_particles):
            guide_trace = poutine.trace(guide,
                                        graph_type="dense").get_trace(*args, **kwargs)
            model_trace = poutine.trace(poutine.replay(model, trace=guide_trace),
                                        graph_type="dense").get_trace(*args, **kwargs)
            if is_validation_enabled():
                check_model_guide_match(model_trace, guide_trace)
                enumerated_sites = [name for name, site in guide_trace.nodes.items()
                                    if site["type"] == "sample" and site["infer"].get("enumerate")]
                if enumerated_sites:
                    warnings.warn('\n'.join([
                        'TraceGraph_ELBO found sample sites configured for enumeration:'
                        ', '.join(enumerated_sites),
                        'If you want to enumerate sites, you need to use TraceEnum_ELBO instead.']))

            guide_trace = prune_subsample_sites(guide_trace)
            model_trace = prune_subsample_sites(model_trace)

            weight = 1.0 / self.num_particles
            yield weight, model_trace, guide_trace

    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for weight, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            elbo_particle = torch_item(model_trace.log_prob_sum()) - torch_item(guide_trace.log_prob_sum())
            elbo += weight * elbo_particle

        loss = -elbo
        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        If baselines are present, a baseline loss is also constructed and differentiated.
        """
        loss = 0.0
        for weight, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
            loss += self._loss_and_grads_particle(weight, model_trace, guide_trace)
        return loss

    def _loss_and_grads_particle(self, weight, model_trace, guide_trace):
        # have the trace compute all the individual (batch) log pdf terms
        # and score function terms (if present) so that they are available below
        model_trace.compute_log_prob()
        guide_trace.compute_score_parts()
        if is_validation_enabled():
            for site in model_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, self.max_iarange_nesting)
            for site in guide_trace.nodes.values():
                if site["type"] == "sample":
                    check_site_shape(site, self.max_iarange_nesting)

        # compute elbo for reparameterized nodes
        non_reparam_nodes = set(guide_trace.nonreparam_stochastic_nodes)
        elbo, surrogate_elbo = _compute_elbo_reparam(model_trace, guide_trace, non_reparam_nodes)

        # the following computations are only necessary if we have non-reparameterizable nodes
        baseline_loss = 0.0
        if non_reparam_nodes:
            downstream_costs, _ = _compute_downstream_costs(model_trace, guide_trace, non_reparam_nodes)
            surrogate_elbo_term, baseline_loss = _compute_elbo_non_reparam(guide_trace,
                                                                           non_reparam_nodes, downstream_costs)
            surrogate_elbo += surrogate_elbo_term

        # collect parameters to train from model and guide
        trainable_params = any(site["type"] == "param"
                               for trace in (model_trace, guide_trace)
                               for site in trace.nodes.values())

        if trainable_params:
            surrogate_loss = -surrogate_elbo
            torch_backward(weight * (surrogate_loss + baseline_loss))

        loss = -torch_item(elbo)
        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return weight * loss


class JitTraceGraph_ELBO(TraceGraph_ELBO):
    """
    Like :class:`TraceGraph_ELBO` but uses :func:`torch.jit.compile` to
    compile :meth:`loss_and_grads`.

    This works only for a limited set of models:

    -   Models must have static structure.
    -   Models must not depend on any global data (except the param store).
    -   All model inputs that are tensors must be passed in via ``*args``.
    -   All model inputs that are *not* tensors must be passed in via
        ``*kwargs``, and these will be fixed to their values on the first
        call to :meth:`loss_and_grads`.

    .. warning:: Experimental. Interface subject to change.
    """

    def loss_and_grads(self, model, guide, *args, **kwargs):
        if getattr(self, '_loss_and_surrogate_loss', None) is None:
            # build a closure for loss_and_surrogate_loss
            weakself = weakref.ref(self)

            @pyro.ops.jit.compile(nderivs=1)
            def loss_and_surrogate_loss(*args):
                self = weakself()
                loss = 0.0
                surrogate_loss = 0.0
                for weight, model_trace, guide_trace in self._get_traces(model, guide, *args, **kwargs):
                    model_trace.compute_log_prob()
                    guide_trace.compute_score_parts()
                    if is_validation_enabled():
                        for site in model_trace.nodes.values():
                            if site["type"] == "sample":
                                check_site_shape(site, self.max_iarange_nesting)
                        for site in guide_trace.nodes.values():
                            if site["type"] == "sample":
                                check_site_shape(site, self.max_iarange_nesting)

                    # compute elbo for reparameterized nodes
                    non_reparam_nodes = set(guide_trace.nonreparam_stochastic_nodes)
                    elbo, surrogate_elbo = _compute_elbo_reparam(model_trace, guide_trace, non_reparam_nodes)

                    # the following computations are only necessary if we have non-reparameterizable nodes
                    baseline_loss = 0.0
                    if non_reparam_nodes:
                        downstream_costs, _ = _compute_downstream_costs(model_trace, guide_trace, non_reparam_nodes)
                        surrogate_elbo_term, baseline_loss = _compute_elbo_non_reparam(guide_trace,
                                                                                       non_reparam_nodes,
                                                                                       downstream_costs)
                        surrogate_elbo += surrogate_elbo_term

                    loss = loss - weight * elbo
                    surrogate_loss = surrogate_loss - weight * surrogate_elbo

                return loss, surrogate_loss

            self._loss_and_surrogate_loss = loss_and_surrogate_loss

        loss, surrogate_loss = self._loss_and_surrogate_loss(*args)
        surrogate_loss.backward()  # this line triggers jit compilation
        loss = loss.item()

        if torch_isnan(loss):
            warnings.warn('Encountered NAN loss')
        return loss
