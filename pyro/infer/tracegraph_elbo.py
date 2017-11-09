from __future__ import absolute_import, division, print_function

import warnings
from collections import namedtuple

import networkx
import torch

import pyro
import pyro.poutine as poutine
from pyro.infer.util import torch_backward, torch_data_sum
from pyro.poutine.util import prune_subsample_sites
from pyro.util import check_model_guide_match, detach_iterable, ng_zeros

CostNode = namedtuple("CostNode", ["cost", "nonzero_expectation"])


def _get_baseline_options(site):
    """
    Extracts baseline options from ``site["baseline"]``.
    """
    # XXX default for baseline_beta currently set here
    options_dict = site["baseline"].copy()
    options_tuple = (options_dict.pop('nn_baseline', None),
                     options_dict.pop('nn_baseline_input', None),
                     options_dict.pop('use_decaying_avg_baseline', False),
                     options_dict.pop('baseline_beta', 0.90),
                     options_dict.pop('baseline_value', None))
    if options_dict:
        raise ValueError("Unrecognized baseline options: {}".format(options_dict.keys()))
    return options_tuple


class TraceGraph_ELBO(object):
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
    def __init__(self, num_particles=1, enum_discrete=False):
        """
        :param num_particles: the number of particles (samples) used to form the estimator
        :param bool enum_discrete: whether to sum over discrete latent variables, rather than sample them
        """
        super(TraceGraph_ELBO, self).__init__()
        self.num_particles = num_particles
        self.enum_discrete = enum_discrete

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a tracegraph generator

        XXX support for automatically settings args/kwargs to volatile?
        """

        for i in range(self.num_particles):
            if self.enum_discrete:
                raise NotImplementedError("https://github.com/uber/pyro/issues/220")

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

            elbo += weight * elbo_particle.data[0]

        loss = -elbo
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
        # get info regarding rao-blackwellization of vectorized map_data
        guide_vec_md_info = guide_trace.graph["vectorized_map_data_info"]
        model_vec_md_info = model_trace.graph["vectorized_map_data_info"]
        guide_vec_md_condition = guide_vec_md_info['rao-blackwellization-condition']
        model_vec_md_condition = model_vec_md_info['rao-blackwellization-condition']
        do_vec_rb = guide_vec_md_condition and model_vec_md_condition
        if not do_vec_rb:
            warnings.warn(
                "Unable to do fully-vectorized Rao-Blackwellization in TraceGraph_ELBO. "
                "Falling back to higher-variance gradient estimator. "
                "Try to avoid these issues in your model and guide:\n{}".format("\n".join(
                    guide_vec_md_info["warnings"] | model_vec_md_info["warnings"])))
        guide_vec_md_nodes = guide_vec_md_info['nodes'] if do_vec_rb else set()
        model_vec_md_nodes = model_vec_md_info['nodes'] if do_vec_rb else set()

        # have the trace compute all the individual (batch) log pdf terms
        # so that they are available below
        guide_trace.compute_batch_log_pdf(site_filter=lambda name, site: name in guide_vec_md_nodes)
        guide_trace.log_pdf()
        model_trace.compute_batch_log_pdf(site_filter=lambda name, site: name in model_vec_md_nodes)
        model_trace.log_pdf()

        # prepare a list of all the cost nodes, each of which is +- log_pdf
        cost_nodes = []
        non_reparam_nodes = set(guide_trace.nonreparam_stochastic_nodes)
        for name, model_site in model_trace.nodes.items():
            if model_site["type"] == "sample":
                if model_site["is_observed"]:
                    cost_nodes.append(CostNode(model_site["log_pdf"], True))
                else:
                    # cost node from model sample
                    cost_nodes.append(CostNode(model_site["log_pdf"], True))
                    # cost node from guide sample
                    guide_site = guide_trace.nodes[name]
                    zero_expectation = name in non_reparam_nodes
                    cost_nodes.append(CostNode(-guide_site["log_pdf"], not zero_expectation))

        # compute the elbo; if all stochastic nodes are reparameterizable, we're done
        # this bit is never differentiated: it's here for getting an estimate of the elbo itself
        elbo = torch_data_sum(sum(c.cost for c in cost_nodes))

        # compute the surrogate elbo, removing terms whose gradient is zero
        # this is the bit that's actually differentiated
        # XXX should the user be able to control if these terms are included?
        surrogate_elbo = sum(c.cost for c in cost_nodes if c.nonzero_expectation)

        # the following computations are only necessary if we have non-reparameterizable nodes
        baseline_loss = 0.0
        if non_reparam_nodes:

            # recursively compute downstream cost nodes for all sample sites in model and guide
            # (even though ultimately just need for non-reparameterizable sample sites)
            # 1. downstream costs used for rao-blackwellization
            # 2. model observe sites (as well as terms that arise from the model and guide having different
            # dependency structures) are taken care of via 'children_in_model' below
            topo_sort_guide_nodes = list(reversed(list(networkx.topological_sort(guide_trace))))
            topo_sort_guide_nodes = [x for x in topo_sort_guide_nodes
                                     if guide_trace.nodes[x]["type"] == "sample"]
            downstream_guide_cost_nodes = {}
            downstream_costs = {}

            for node in topo_sort_guide_nodes:
                node_log_pdf_key = 'batch_log_pdf' if node in guide_vec_md_nodes else 'log_pdf'
                downstream_costs[node] = model_trace.nodes[node][node_log_pdf_key] - \
                    guide_trace.nodes[node][node_log_pdf_key]
                nodes_included_in_sum = set([node])
                downstream_guide_cost_nodes[node] = set([node])
                for child in guide_trace.successors(node):
                    child_cost_nodes = downstream_guide_cost_nodes[child]
                    downstream_guide_cost_nodes[node].update(child_cost_nodes)
                    if nodes_included_in_sum.isdisjoint(child_cost_nodes):  # avoid duplicates
                        if node_log_pdf_key == 'log_pdf':
                            downstream_costs[node] += downstream_costs[child].sum()
                        else:
                            downstream_costs[node] += downstream_costs[child]
                        nodes_included_in_sum.update(child_cost_nodes)
                missing_downstream_costs = downstream_guide_cost_nodes[node] - nodes_included_in_sum
                # include terms we missed because we had to avoid duplicates
                for missing_node in missing_downstream_costs:
                    mn_log_pdf_key = 'batch_log_pdf' if missing_node in guide_vec_md_nodes else 'log_pdf'
                    if node_log_pdf_key == 'log_pdf':
                        downstream_costs[node] += (model_trace.nodes[missing_node][mn_log_pdf_key] -
                                                   guide_trace.nodes[missing_node][mn_log_pdf_key]).sum()
                    else:
                        downstream_costs[node] += model_trace.nodes[missing_node][mn_log_pdf_key] - \
                                                  guide_trace.nodes[missing_node][mn_log_pdf_key]

            # finish assembling complete downstream costs
            # (the above computation may be missing terms from model)
            # XXX can we cache some of the sums over children_in_model to make things more efficient?
            for site in non_reparam_nodes:
                children_in_model = set()
                for node in downstream_guide_cost_nodes[site]:
                    children_in_model.update(model_trace.successors(node))
                # remove terms accounted for above
                children_in_model.difference_update(downstream_guide_cost_nodes[site])
                for child in children_in_model:
                    child_log_pdf_key = 'batch_log_pdf' if child in model_vec_md_nodes else 'log_pdf'
                    site_log_pdf_key = 'batch_log_pdf' if site in guide_vec_md_nodes else 'log_pdf'
                    assert (model_trace.nodes[child]["type"] == "sample")
                    if site_log_pdf_key == 'log_pdf':
                        downstream_costs[site] += model_trace.nodes[child][child_log_pdf_key].sum()
                    else:
                        downstream_costs[site] += model_trace.nodes[child][child_log_pdf_key]

            # construct all the reinforce-like terms.
            # we include only downstream costs to reduce variance
            # optionally include baselines to further reduce variance
            # XXX should the average baseline be in the param store as below?
            elbo_reinforce_terms = 0.0
            for node in non_reparam_nodes:
                guide_site = guide_trace.nodes[node]
                log_pdf_key = 'batch_log_pdf' if node in guide_vec_md_nodes else 'log_pdf'
                downstream_cost = downstream_costs[node]
                baseline = 0.0
                (nn_baseline, nn_baseline_input, use_decaying_avg_baseline, baseline_beta,
                    baseline_value) = _get_baseline_options(guide_site)
                use_nn_baseline = nn_baseline is not None
                use_baseline_value = baseline_value is not None
                assert(not (use_nn_baseline and use_baseline_value)), \
                    "cannot use baseline_value and nn_baseline simultaneously"
                if use_decaying_avg_baseline:
                    avg_downstream_cost_old = pyro.param("__baseline_avg_downstream_cost_" + node,
                                                         ng_zeros(1), tags="__tracegraph_elbo_internal_tag")
                    avg_downstream_cost_new = (1 - baseline_beta) * downstream_cost + \
                        baseline_beta * avg_downstream_cost_old
                    avg_downstream_cost_old.data = avg_downstream_cost_new.data  # XXX copy_() ?
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

                guide_log_pdf = guide_site[log_pdf_key] / guide_site["scale"]  # not scaled by subsampling
                if use_nn_baseline or use_decaying_avg_baseline or use_baseline_value:
                    if downstream_cost.size() != baseline.size():
                        raise ValueError("Expected baseline at site {} to be {} instead got {}".format(
                            node, downstream_cost.size(), baseline.size()))
                    downstream_cost = downstream_cost - baseline
                elbo_reinforce_terms += (guide_log_pdf * downstream_cost.detach()).sum()

            surrogate_elbo += elbo_reinforce_terms

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
        return weight * loss
