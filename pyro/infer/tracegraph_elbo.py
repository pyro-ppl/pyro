import networkx
import torch

import pyro
import pyro.poutine as poutine
from pyro.util import ng_zeros, detach_iterable


class TraceGraph_ELBO(object):
    """
    A TraceGraph implementation of ELBO-based SVI
    The gradient estimator is constructed along the lines of

    'Gradient Estimation Using Stochastic Computation Graphs'
    John Schulman, Nicolas Heess, Theophane Weber, Pieter Abbeel

    specialized to the case of the ELBO. It supports arbitrary
    dependency structure for the model and guide. Where possible,
    dependency information as recorded in the TraceGraph is used
    to reduce the variance of the gradient estimator.
    """
    def __init__(self, num_particles=1):
        """
        :param num_particles: the number of particles (samples) used to form the estimator
        """
        super(TraceGraph_ELBO, self).__init__()
        self.num_particles = num_particles

    def _get_traces(self, model, guide, *args, **kwargs):
        """
        runs the guide and runs the model against the guide with
        the result packaged as a tracegraph generator

        XXX support for automatically settings args/kwargs to volatile?
        """

        for i in range(self.num_particles):
            guide_tracegraph = poutine.tracegraph(guide).get_trace(*args, **kwargs)
            guide_trace = guide_tracegraph.get_trace()
            model_tracegraph = poutine.tracegraph(poutine.replay(model, guide_trace)).get_trace(*args, **kwargs)
            yield model_tracegraph, guide_tracegraph

    def loss(self, model, guide, *args, **kwargs):
        """
        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        :returns: returns an estimate of the ELBO
        :rtype: float
        """
        elbo = 0.0
        for model_tracegraph, guide_tracegraph in self._get_traces(model, guide, *args, **kwargs):
            guide_trace, model_trace = guide_tracegraph.get_trace(), model_tracegraph.get_trace()
            guide_trace.log_pdf(), model_trace.log_pdf()

            elbo_particle = 0.0

            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    elbo_particle += model_trace[name]["log_pdf"]
                elif model_trace[name]["type"] == "sample":
                    elbo_particle += model_trace[name]["log_pdf"]
                    elbo_particle -= guide_trace[name]["log_pdf"]

            elbo += elbo_particle.data[0] / self.num_particles

        loss = -elbo
        return loss

    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        If baselines are present, a baseline loss is also constructed and differentiated.
        :returns: returns an estimate of the ELBO
        :rtype: float
        """
        elbo = 0.0
        trainable_params = set()

        for model_tracegraph, guide_tracegraph in self._get_traces(model, guide, *args, **kwargs):
            guide_trace, model_trace = guide_tracegraph.get_trace(), model_tracegraph.get_trace()

            # get info regarding rao-blackwellization of vectorized map_data
            guide_vec_md_info = guide_tracegraph.vectorized_map_data_info
            model_vec_md_info = model_tracegraph.vectorized_map_data_info
            guide_vec_md_condition = guide_vec_md_info['rao-blackwellization-condition']
            model_vec_md_condition = model_vec_md_info['rao-blackwellization-condition']
            do_vec_rb = guide_vec_md_condition and model_vec_md_condition
            guide_vec_md_nodes = guide_vec_md_info['nodes'] if do_vec_rb else None
            model_vec_md_nodes = model_vec_md_info['nodes'] if do_vec_rb else None

            def get_vec_batch_nodes_dict(vec_batch_nodes):
                vec_batch_nodes = vec_batch_nodes.values() if vec_batch_nodes is not None else []
                vec_batch_nodes = [item for sublist in vec_batch_nodes for item in sublist]
                vec_batch_nodes_dict = {}
                for pair in vec_batch_nodes:
                    vec_batch_nodes_dict[pair[0]] = pair[1]
                return vec_batch_nodes_dict

            # these dictionaries encode which sites are batched
            guide_vec_batch_nodes_dict = get_vec_batch_nodes_dict(guide_vec_md_nodes)
            model_vec_batch_nodes_dict = get_vec_batch_nodes_dict(model_vec_md_nodes)

            # have the trace compute all the individual (batch) log pdf terms
            # so that they are available below
            guide_trace.log_pdf(vec_batch_nodes_dict=guide_vec_batch_nodes_dict)
            model_trace.log_pdf(vec_batch_nodes_dict=model_vec_batch_nodes_dict)

            # prepare a list of all the cost nodes, each of which is +- log_pdf
            cost_nodes = []
            non_reparam_nodes = set(guide_tracegraph.get_nonreparam_stochastic_nodes())
            for site in model_trace.keys():
                model_trace_site = model_trace[site]
                log_pdf_key = 'log_pdf' if site not in model_vec_batch_nodes_dict else 'batch_log_pdf'
                if model_trace_site["type"] == "observe":
                    cost_node = (model_trace_site[log_pdf_key], True)
                    cost_nodes.append(cost_node)
                elif model_trace_site["type"] == "sample":
                    # cost node from model sample
                    cost_node1 = (model_trace_site[log_pdf_key], True)
                    # cost node from guide sample
                    zero_expectation = site in non_reparam_nodes
                    cost_node2 = (-guide_trace[site][log_pdf_key],
                                  not zero_expectation)
                    cost_nodes.extend([cost_node1, cost_node2])

            elbo_particle = 0.0
            surrogate_elbo_particle = 0.0
            baseline_loss_particle = 0.0
            elbo_reinforce_terms_particle = 0.0
            elbo_no_zero_expectation_terms_particle = 0.0

            # compute the elbo; if all stochastic nodes are reparameterizable, we're done
            # this bit is never differentiated: it's here for getting an estimate of the elbo itself
            for cost_node in cost_nodes:
                elbo_particle += cost_node[0].sum()
            elbo += elbo_particle.data[0] / self.num_particles

            # compute the elbo, removing terms whose gradient is zero
            # this is the bit that's actually differentiated
            # XXX should the user be able to control if these terms are included?
            for cost_node in cost_nodes:
                if cost_node[1]:
                    elbo_no_zero_expectation_terms_particle += cost_node[0].sum()
            surrogate_elbo_particle += elbo_no_zero_expectation_terms_particle / self.num_particles

            # the following computations are only necessary if we have non-reparameterizable nodes
            if len(non_reparam_nodes) > 0:

                # recursively compute downstream cost nodes for all sample sites in model and guide
                # (even though ultimately just need for non-reparameterizable sample sites)
                # 1. downstream costs used for rao-blackwellization
                # 2. model observe sites (as well as terms that arise from the model and guide having different
                # dependency structures) are taken care of via 'children_in_model' below
                guide_dag = guide_tracegraph.get_graph()
                topo_sort_guide_nodes = list(reversed(list(networkx.topological_sort(guide_dag))))
                downstream_guide_cost_nodes = {}
                downstream_costs = {}

                for node in topo_sort_guide_nodes:
                    node_log_pdf_key = 'log_pdf' if node not in guide_vec_batch_nodes_dict else 'batch_log_pdf'
                    downstream_costs[node] = model_trace[node][node_log_pdf_key] - \
                        guide_trace[node][node_log_pdf_key]
                    nodes_included_in_sum = set([node])
                    downstream_guide_cost_nodes[node] = set([node])
                    for child in guide_tracegraph.get_children(node):
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
                        mn_log_pdf_key = 'log_pdf' if missing_node not in \
                                            guide_vec_batch_nodes_dict else 'batch_log_pdf'
                        if node_log_pdf_key == 'log_pdf':
                            downstream_costs[node] += (model_trace[missing_node][mn_log_pdf_key] -
                                                       guide_trace[missing_node][mn_log_pdf_key]).sum()
                        else:
                            downstream_costs[node] += model_trace[missing_node][mn_log_pdf_key] - \
                                                      guide_trace[missing_node][mn_log_pdf_key]

                # finish assembling complete downstream costs
                # (the above computation may be missing terms from model)
                # XXX can we cache some of the sums over children_in_model to make things more efficient?
                for site in non_reparam_nodes:
                    children_in_model = set()
                    for node in downstream_guide_cost_nodes[site]:
                        children_in_model.update(model_tracegraph.get_children(node))
                    # remove terms accounted for above
                    children_in_model.difference_update(downstream_guide_cost_nodes[site])
                    for child in children_in_model:
                        child_log_pdf_key = 'log_pdf' if child not in model_vec_batch_nodes_dict \
                            else 'batch_log_pdf'
                        site_log_pdf_key = 'log_pdf' if site not in guide_vec_batch_nodes_dict \
                            else 'batch_log_pdf'
                        assert (model_trace[child]["type"] in ("sample", "observe"))
                        if site_log_pdf_key == 'log_pdf':
                            downstream_costs[site] += model_trace[child][child_log_pdf_key].sum()
                        else:
                            downstream_costs[site] += model_trace[child][child_log_pdf_key]

                # construct all the reinforce-like terms.
                # we include only downstream costs to reduce variance
                # optionally include baselines to further reduce variance
                # XXX should the average baseline be in the param store as below?

                # for extracting baseline options from kwargs
                # XXX default for baseline_beta currently set here
                def get_baseline_kwargs(kwargs):
                    return kwargs.get('nn_baseline', None), \
                           kwargs.get('nn_baseline_input', None), \
                           kwargs.get('use_decaying_avg_baseline', False), \
                           kwargs.get('baseline_beta', 0.90), \
                           kwargs.get('baseline_value', None)

                baseline_loss_particle = 0.0
                for node in non_reparam_nodes:
                    log_pdf_key = 'log_pdf' if node not in guide_vec_batch_nodes_dict else 'batch_log_pdf'
                    downstream_cost = downstream_costs[node]
                    baseline = 0.0
                    nn_baseline, nn_baseline_input, use_decaying_avg_baseline, baseline_beta, \
                        baseline_value = get_baseline_kwargs(guide_trace[node]['args'][1])
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
                        # construct baseline loss
                        baseline_loss = torch.pow(downstream_cost.detach() - baseline, 2.0).sum()
                        baseline_loss_particle += baseline_loss / self.num_particles
                    if use_nn_baseline or use_decaying_avg_baseline or use_baseline_value:
                        elbo_reinforce_terms_particle += (guide_trace[node][log_pdf_key] *
                                                          (downstream_cost - baseline).detach()).sum()
                    else:
                        elbo_reinforce_terms_particle += (guide_trace[node][log_pdf_key] *
                                                          downstream_cost.detach()).sum()

                surrogate_elbo_particle += elbo_reinforce_terms_particle / self.num_particles
                if not isinstance(baseline_loss_particle, float):
                    baseline_loss_particle.backward()

            # grab model parameters to train
            for name in model_trace.keys():
                if model_trace[name]["type"] == "param":
                    trainable_params.add(model_trace[name]["value"])

            # grab guide parameters to train
            for name in guide_trace.keys():
                if guide_trace[name]["type"] == "param":
                    trainable_params.add(guide_trace[name]["value"])

            # mark all params seen in trace as active so that gradient steps are taken downstream
            pyro.get_param_store().mark_params_active(trainable_params)

            surrogate_loss_particle = -surrogate_elbo_particle
            surrogate_loss_particle.backward()

        loss = -elbo

        return loss
