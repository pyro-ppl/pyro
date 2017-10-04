import networkx
import torch

import pyro
import pyro.poutine as poutine
from pyro.util import ng_zeros, detach_iterable


class TraceGraph_ELBO(object):
    """
    A TraceGraph and Poutine-based implementation of SVI
    The gradient estimator is constructed along the lines of
    'Gradient Estimation Using Stochastic Computation Graphs'
    John Schulman, Nicolas Heess, Theophane Weber, Pieter Abbeel

    :param model: the model (callable)
    :param guide: the guide (callable), i.e. the variational distribution
    :param num_particles: the number of particles (samples) used to form the estimator
    """

    def __init__(self, model, guide, num_particles=1, *args, **kwargs):
        # initialize
        super(TraceGraph_ELBO, self).__init__()
        self.model = model
        self.guide = guide
        self.num_particles = num_particles

    def _get_traces(self, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a tracegraph generator
        """

        for i in range(self.num_particles):
            guide_tracegraph = poutine.tracegraph(self.guide)(*args, **kwargs)
            guide_trace = guide_tracegraph.get_trace()
            model_tracegraph = poutine.tracegraph(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)
            yield model_tracegraph, guide_tracegraph

    def loss(self, *args, **kwargs):
        """
        Evaluate Elbo by running num_particles often.
        Returns the Elbo as a value
        """
        elbo = 0.0
        for model_tracegraph, guide_tracegraph in self._get_traces(*args, **kwargs):
            guide_trace, model_trace = guide_tracegraph.get_trace(), model_tracegraph.get_trace()
            guide_trace.log_pdf(), model_trace.log_pdf()

            elbo_particle = 0.0

            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    elbo_particle += model_trace[name]["log_pdf"]
                elif model_trace[name]["type"] == "sample":
                    elbo_particle += model_trace[name]["log_pdf"]
                    elbo_particle -= guide_trace[name]["log_pdf"]

            elbo += elbo_particle / self.num_particles

        loss = -elbo
        return loss.data[0]

    def loss_and_grads(self, *args, **kwargs):
        """
        computes the elbo as well as the surrogate elbo. performs backward on latter.
        num_particle many samples are used to form the estimators.
        returns an estimate of the elbo as well as the trainable_params_dict.
        implicitly returns gradients via param.grad for each param in the trainable_params_dict.
        """
        elbo = 0.0
        surrogate_elbo = 0.0
        baseline_loss = 0.0
        trainable_params_dict = {}
        baseline_params = set()

        def add_to_trainable_params_dict(param_name, param_value):
            if param_name not in trainable_params_dict:
                trainable_params_dict[param_name] = param_value

        for model_tracegraph, guide_tracegraph in self._get_traces(*args, **kwargs):
            guide_trace, model_trace = guide_tracegraph.get_trace(), model_tracegraph.get_trace()

            # have the trace compute all the individual log pdf terms
            # so that they are available below
            guide_trace.log_pdf(), model_trace.log_pdf()

            # prepare a list of all the cost nodes, each of which is +- log_pdf
            cost_nodes = []
            non_reparam_nodes = set(guide_tracegraph.get_nonreparam_stochastic_nodes())
            for site in model_trace.keys():
                model_trace_site = model_trace[site]
                if model_trace_site["type"] == "observe":
                    cost_node = (model_trace_site["log_pdf"], True)
                    cost_nodes.append(cost_node)
                elif model_trace_site["type"] == "sample":
                    # cost node from model sample
                    cost_node1 = (model_trace_site["log_pdf"], True)
                    # cost node from guide sample
                    zero_expectation = site in non_reparam_nodes
                    cost_node2 = (-guide_trace[site]["log_pdf"],
                                  not zero_expectation)
                    cost_nodes.extend([cost_node1, cost_node2])

            elbo_particle, elbo_reinforce_terms_particle = 0.0, 0.0
            elbo_no_zero_expectation_terms_particle = 0.0

            # compute the elbo; if all stochastic nodes are reparameterizable, we're done
            # this bit is never differentiated: it's here for getting an estimate of the elbo itself
            for cost_node in cost_nodes:
                elbo_particle += cost_node[0]
            elbo += elbo_particle / self.num_particles

            # compute the elbo, removing terms whose gradient is zero
            # this is the bit that's actually differentiated
            for cost_node in cost_nodes:
                if cost_node[1]:
                    elbo_no_zero_expectation_terms_particle += cost_node[0]

            # the following computations are only necessary if we have non-reparameterizable nodes
            if len(non_reparam_nodes) > 0:

                # recursively compute downstream cost nodes for all sample sites in model and guide
                # (even though ultimately just need for non-reparameterizable sample sites)
                # 1. downstream costs used for rao-blackwellization
                # 2. model observe sites (as well as terms that arise from the model and guide having different
                # dependency structures) are taken care of via 'children_in_model' below
                guide_dag = guide_tracegraph.get_graph()
                topo_sort_guide_nodes = list(
                    reversed(list(networkx.topological_sort(guide_dag))))
                downstream_guide_cost_nodes = {}
                downstream_costs = {}

                for node in topo_sort_guide_nodes:
                    downstream_costs[
                        node] = model_trace[node]["log_pdf"] - guide_trace[node]["log_pdf"]
                    nodes_included_in_sum = set([node])
                    downstream_guide_cost_nodes[node] = set([node])
                    for child in guide_tracegraph.get_children(node):
                        child_cost_nodes = downstream_guide_cost_nodes[child]
                        downstream_guide_cost_nodes[node].update(
                            child_cost_nodes)
                        if nodes_included_in_sum.isdisjoint(
                                child_cost_nodes):  # avoid duplicates
                            downstream_costs[node] += downstream_costs[child]
                            nodes_included_in_sum.update(child_cost_nodes)
                    missing_downstream_costs = downstream_guide_cost_nodes[node] - nodes_included_in_sum
                    # include terms we missed because we had to avoid duplicates
                    for missing_node in missing_downstream_costs:
                        downstream_costs[node] += model_trace[missing_node]["log_pdf"] - \
                             guide_trace[missing_node]["log_pdf"]

                # finish assembling complete downstream costs
                # (the above computation may be missing terms from model)
                # XXX can we cache some of the sums over children_in_model to make things more efficient?
                for site in non_reparam_nodes:
                    children_in_model = set()
                    for node in downstream_guide_cost_nodes[site]:
                        children_in_model.update(
                            model_tracegraph.get_children(node))
                    # remove terms accounted for above
                    children_in_model.difference_update(
                        downstream_guide_cost_nodes[site])
                    for child in children_in_model:
                        assert (model_trace[child]["type"] in ("sample",
                                                               "observe"))
                        downstream_costs[site] += model_trace[child]["log_pdf"]

                # construct all the reinforce-like terms.
                # we include only downstream costs to reduce variance
                # optionally include baselines to further reduce variance
                # XXX should the average baseline be in the param store as below?

                # for extracting baseline options from kwargs
                def get_baseline_kwargs(kwargs):
                    return kwargs.get('nn_baseline', None), \
                        kwargs.get('nn_baseline_input', None), \
                        kwargs.get('use_decaying_avg_baseline', False), \
                        kwargs.get('baseline_beta', 0.90)  # default decay rate for avg_baseline

                # this [] will be used to store information need to construct baseline losses below
                baseline_losses_particle = []
                for node in non_reparam_nodes:
                    downstream_cost = downstream_costs[node]
                    baseline = 0.0
                    nn_baseline, nn_baseline_input, use_decaying_avg_baseline, \
                        baseline_beta = get_baseline_kwargs(guide_trace[node]['args'][1])
                    use_nn_baseline = nn_baseline is not None
                    if use_decaying_avg_baseline:
                        avg_downstream_cost_old = pyro.param(
                            "__baseline_avg_downstream_cost_" + node,
                            ng_zeros(1))
                        avg_downstream_cost_new = (1 - baseline_beta) * downstream_cost + \
                            baseline_beta * avg_downstream_cost_old
                        avg_downstream_cost_old.data = avg_downstream_cost_new.data  # XXX copy_() ?
                        baseline += avg_downstream_cost_old
                    if use_nn_baseline:
                        # block nn_baseline_input gradients except in baseline loss
                        baseline += nn_baseline(
                            detach_iterable(nn_baseline_input))
                        nn_params = nn_baseline.parameters()
                        baseline_loss_particle = torch.pow(
                            downstream_cost.detach() - baseline, 2.0)
                        baseline_losses_particle.append(
                            (baseline_loss_particle, nn_params))
                    if use_nn_baseline or use_decaying_avg_baseline:
                        elbo_reinforce_terms_particle += guide_trace[node]['log_pdf'] * \
                         (downstream_cost - baseline.data).detach()
                    else:
                        elbo_reinforce_terms_particle += guide_trace[node]['log_pdf'] * \
                         downstream_cost.detach()

                for _loss, _params in baseline_losses_particle:
                    baseline_loss += _loss / self.num_particles
                    baseline_params.update(_params)

                surrogate_elbo += elbo_no_zero_expectation_terms_particle / self.num_particles
                surrogate_elbo += elbo_reinforce_terms_particle / self.num_particles

                # grab model parameters to train
                for name in model_trace.keys():
                    if model_trace[name]["type"] == "param":
                        add_to_trainable_params_dict(name, model_trace[name]["value"])

                # grab guide parameters to train
                for name in guide_trace.keys():
                    if guide_trace[name]["type"] == "param":
                        add_to_trainable_params_dict(name, guide_trace[name]["value"])

            surrogate_loss = -surrogate_elbo
            surrogate_loss.backward()
            loss = -elbo

            return loss, trainable_params_dict, baseline_loss, baseline_params
