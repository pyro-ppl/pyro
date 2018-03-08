from __future__ import absolute_import, division, print_function

import warnings
from operator import itemgetter

import networkx
import torch
from torch.autograd import variable

import pyro
import pyro.poutine as poutine
from pyro.distributions.util import is_identically_zero
from pyro.infer import ELBO
from pyro.infer.util import MultiViewTensor as MVT
from pyro.infer.util import torch_backward, torch_data_sum
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, check_site_shape, detach_iterable, is_nan


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
                              model_iarange_nodes, guide_iarange_nodes,  #
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
    stacks = model_trace.graph["iarange_info"]['iarange_stacks']

    def n_compatible_indices(dest_node, source_node):
        n_compatible = 0
        for xframe, yframe in zip(stacks[source_node], stacks[dest_node]):
            if xframe.name == yframe.name:
                n_compatible += 1
        return n_compatible

    for node in topo_sort_guide_nodes:
        downstream_costs[node] = MVT(model_trace.nodes[node]['batch_log_pdf'] -
                                     guide_trace.nodes[node]['batch_log_pdf'])
        nodes_included_in_sum = set([node])
        downstream_guide_cost_nodes[node] = set([node])
        # make more efficient by ordering children appropriately (higher children first)
        children = [(k, -ordered_guide_nodes_dict[k]) for k in guide_trace.successors(node)]
        sorted_children = sorted(children, key=itemgetter(1))
        for child, _ in sorted_children:
            child_cost_nodes = downstream_guide_cost_nodes[child]
            downstream_guide_cost_nodes[node].update(child_cost_nodes)
            if nodes_included_in_sum.isdisjoint(child_cost_nodes):  # avoid duplicates
                dims_to_keep = n_compatible_indices(node, child)
                summed_child = downstream_costs[child].sum_leftmost_all_but(dims_to_keep)
                downstream_costs[node].add(summed_child)
                # XXX nodes_included_in_sum logic could be more fine-grained, possibly leading
                # to speed-ups in case there are many duplicates
                nodes_included_in_sum.update(child_cost_nodes)
        missing_downstream_costs = downstream_guide_cost_nodes[node] - nodes_included_in_sum
        # include terms we missed because we had to avoid duplicates
        for missing_node in missing_downstream_costs:
            missing_term = MVT(model_trace.nodes[missing_node]['batch_log_pdf'] -
                               guide_trace.nodes[missing_node]['batch_log_pdf'])
            dims_to_keep = n_compatible_indices(node, missing_node)
            summed_missing_term = missing_term.sum_leftmost_all_but(dims_to_keep)
            downstream_costs[node].add(summed_missing_term)

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
            dims_to_keep = n_compatible_indices(site, child)
            summed_child = MVT(model_trace.nodes[child]['batch_log_pdf']).sum_leftmost_all_but(dims_to_keep)
            downstream_costs[site].add(summed_child)
            downstream_guide_cost_nodes[site].update([child])

    for k in topo_sort_guide_nodes:
        downstream_costs[k] = downstream_costs[k].contract_as(guide_trace.nodes[k]['batch_log_pdf'])

    return downstream_costs, downstream_guide_cost_nodes


def _compute_elbo_reparam(model_trace, guide_trace, non_reparam_nodes):
    elbo = 0.0
    surrogate_elbo = 0.0
    for name, model_site in model_trace.nodes.items():
        if model_site["type"] == "sample":
            if model_site["is_observed"]:
                elbo += model_site["log_pdf"]
                surrogate_elbo += model_site["log_pdf"]
            else:
                # deal with log p(z|...) term
                elbo += model_site["log_pdf"]
                surrogate_elbo += model_site["log_pdf"]
                # deal with log q(z|...) term, if present
                guide_site = guide_trace.nodes[name]
                elbo -= guide_site["log_pdf"]
                entropy_term = guide_site["score_parts"].entropy_term
                if not is_identically_zero(entropy_term):
                    surrogate_elbo -= entropy_term.sum()

    # elbo is never differentiated, surrogate_elbo is

    return torch_data_sum(elbo), surrogate_elbo


def _compute_elbo_non_reparam(guide_trace, guide_iarange_nodes,  #
                              non_reparam_nodes, downstream_costs):
    # construct all the reinforce-like terms.
    # we include only downstream costs to reduce variance
    # optionally include baselines to further reduce variance
    # XXX should the average baseline be in the param store as below?
    surrogate_elbo = 0.0
    baseline_loss = 0.0
    for node in non_reparam_nodes:
        guide_site = guide_trace.nodes[node]
        log_pdf_key = 'batch_log_pdf' if node in guide_iarange_nodes else 'log_pdf'
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
            avg_downstream_cost_old = pyro.param("__baseline_avg_downstream_cost_" + node,
                                                 variable(0.0).expand(dc_shape).clone(),
                                                 tags="__tracegraph_elbo_internal_tag")
            avg_downstream_cost_new = (1 - baseline_beta) * downstream_cost.detach() + \
                baseline_beta * avg_downstream_cost_old
            avg_downstream_cost_old.copy_(avg_downstream_cost_new)  # XXX is this copy_() what we want?
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
        if log_pdf_key == 'log_pdf':
            score_function_term = score_function_term.sum()
        if use_nn_baseline or use_decaying_avg_baseline or use_baseline_value:
            if downstream_cost.size() != baseline.size():
                raise ValueError("Expected baseline at site {} to be {} instead got {}".format(
                    node, downstream_cost.size(), baseline.size()))
            downstream_cost = downstream_cost - baseline
        surrogate_elbo += (score_function_term * downstream_cost.detach()).sum()

    return surrogate_elbo, baseline_loss


class TraceGraph_ELBO(ELBO):
    """
    A TraceGraph implementation of ELBO-based SVI. The gradient estimator
    is constructed along the lines of reference [1] specialized to the case
    of the ELBO. It supports arbitrary dependency structure for the model
    and guide as well as baselines for non-reparameteriable random variables.
    Where possible, dependency information as recorded in the TraceGraph is
    used to reduce the variance of the gradient estimator.

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
            model_trace = poutine.trace(poutine.replay(model, guide_trace),
                                        graph_type="dense").get_trace(*args, **kwargs)

            check_model_guide_match(model_trace, guide_trace)
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
            guide_trace.log_pdf(), model_trace.log_pdf()

            elbo_particle = 0.0

            for name in model_trace.nodes.keys():
                if model_trace.nodes[name]["type"] == "sample":
                    if model_trace.nodes[name]["is_observed"]:
                        elbo_particle += model_trace.nodes[name]["log_pdf"]
                    else:
                        elbo_particle += model_trace.nodes[name]["log_pdf"]
                        elbo_particle -= guide_trace.nodes[name]["log_pdf"]

            elbo += torch_data_sum(weight * elbo_particle)

        loss = -elbo
        if is_nan(loss):
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
        # get info regarding rao-blackwellization of iarange
        guide_iarange_nodes = guide_trace.graph["iarange_info"]['nodes']
        model_iarange_nodes = model_trace.graph["iarange_info"]['nodes']

        # have the trace compute all the individual (batch) log pdf terms
        # and score function terms (if present) so that they are available below
        model_trace.compute_batch_log_pdf()
        for site in model_trace.nodes.values():
            if site["type"] == "sample":
                check_site_shape(site, self.max_iarange_nesting)
        guide_trace.compute_score_parts()
        for site in guide_trace.nodes.values():
            if site["type"] == "sample":
                check_site_shape(site, self.max_iarange_nesting)

        # compute elbo for reparameterized nodes
        non_reparam_nodes = set(guide_trace.nonreparam_stochastic_nodes)
        elbo, surrogate_elbo = _compute_elbo_reparam(model_trace, guide_trace, non_reparam_nodes)

        # the following computations are only necessary if we have non-reparameterizable nodes
        baseline_loss = 0.0
        if non_reparam_nodes:
            downstream_costs, _ = _compute_downstream_costs(
                    model_trace, guide_trace,  model_iarange_nodes, guide_iarange_nodes, non_reparam_nodes)
            surrogate_elbo_term, baseline_loss = _compute_elbo_non_reparam(
                    guide_trace, guide_iarange_nodes, non_reparam_nodes, downstream_costs)
            surrogate_elbo += surrogate_elbo_term

        # collect parameters to train from model and guide
        trainable_params = set(site["value"]
                               for trace in (model_trace, guide_trace)
                               for site in trace.nodes.values()
                               if site["type"] == "param")

        if trainable_params:
            surrogate_loss = -surrogate_elbo
            torch_backward(weight * (surrogate_loss + baseline_loss))
            pyro.get_param_store().mark_params_active(trainable_params)

        loss = -elbo
        if is_nan(loss):
            warnings.warn('Encountered NAN loss')
        return weight * loss
