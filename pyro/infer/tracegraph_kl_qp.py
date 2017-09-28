import networkx
import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine
from pyro.util import ng_zeros, detach_iterable


class TraceGraph_KL_QP(object):
    """
    A TraceGraph and Poutine-based implementation of SVI
    The basic structure gradient estimator is constructed along the lines of

    'Gradient Estimation Using Stochastic Computation Graphs'
    John Schulman, Nicolas Heess, Theophane Weber, Pieter Abbeel

    :param mu:
    :param model: the model (callable)
    :param guide: the guide (callable), i.e. the variational distribution
    :param optim_step_fct: the pyro.optim object that is used to take a gradient
    step maximizing the ELBO
    :param model_fixed: optional flag which freezes the parameters in the model
    :param guide_fixed: optional flag which freezes the parameters in the guide
    :param optim_step_fct: [optional] pyro.optim object that is used to take gradient
    steps on the baseline losses; defaults to `optim_step_fct`

    """

    def __init__(self,
                 model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False,
                 baseline_optim_step_fct=None,
                 *args, **kwargs):
        """
        Call parent class initially, then setup the poutines to run
        """
        # initialize
        super(TraceGraph_KL_QP, self).__init__()
        # TODO init this somewhere else in a more principled way
        self.model = model
        self.guide = guide
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed
        self.baseline_optim_step_fct = baseline_optim_step_fct

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        single step?
        """
        # call guide() and model() and record trace/tracegraph
        guide_tracegraph = poutine.tracegraph(self.guide)(*args, **kwargs)
        guide_trace = guide_tracegraph.get_trace()
        model_tracegraph = poutine.tracegraph(poutine.replay(self.model, guide_trace))(*args, **kwargs)
        model_trace = model_tracegraph.get_trace()

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
                cost_node2 = (- guide_trace[site][log_pdf_key], not zero_expectation)
                cost_nodes.extend([cost_node1, cost_node2])

        elbo, elbo_reinforce_terms, elbo_no_zero_expectation_terms = 0.0, 0.0, 0.0

        # compute the elbo; if all stochastic nodes are reparameterizable, we're done
        # this bit is never differentiated: it's here for getting an estimate of the elbo itself
        for cost_node in cost_nodes:
            elbo += cost_node[0].sum()

        # compute the elbo, removing terms whose gradient is zero
        # this is the bit that's actually differentiated
        for cost_node in cost_nodes:
            if cost_node[1]:
                elbo_no_zero_expectation_terms += cost_node[0].sum()

        # the following computations are only necessary if we have non-reparameterizable nodes
        if len(non_reparam_nodes) > 0:

            # recursively compute downstream cost nodes for all sample sites in model and guide
            # (even though ultimately just need for non-reparameterizable sample sites)
            # 1. downstream costs used for rao-blackwellization
            # 2. model observe sites (as well as terms that arise from the model and guide having different
            # dependency structures) are taken care of via 'children_in_model' below
            topo_sort_guide_nodes = list(reversed(list(networkx.topological_sort(guide_tracegraph.get_graph()))))
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
                    missing_node_log_pdf_key = 'log_pdf' if missing_node not in guide_vec_batch_nodes_dict \
                        else 'batch_log_pdf'
                    if node_log_pdf_key == 'log_pdf':
                        downstream_costs[node] += (model_trace[missing_node][missing_node_log_pdf_key] -
                                                   guide_trace[missing_node][missing_node_log_pdf_key]).sum()
                    else:
                        downstream_costs[node] += model_trace[missing_node][missing_node_log_pdf_key] - \
                                                  guide_trace[missing_node][missing_node_log_pdf_key]

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
                       kwargs.get('baseline_value', None), \
                       kwargs.get('baseline_params', None)

            # this [] will be used to store information need to construct baseline loss below
            baseline_losses = []
            for node in non_reparam_nodes:
                log_pdf_key = 'log_pdf' if node not in guide_vec_batch_nodes_dict else 'batch_log_pdf'
                downstream_cost = downstream_costs[node]
                baseline = ng_zeros(1)
                nn_baseline, nn_baseline_input, use_decaying_avg_baseline, baseline_beta, \
                    baseline_value, baseline_params = get_baseline_kwargs(guide_trace[node]['args'][1])
                use_nn_baseline = nn_baseline is not None
                use_baseline_value = baseline_value is not None
                assert(not (use_nn_baseline and use_baseline_value)), \
                    "cannot use baseline_value and nn_baseline simultaneously"
                if use_decaying_avg_baseline:
                    avg_downstream_cost_old = pyro.param("__baseline_avg_downstream_cost_" + node,
                                                         ng_zeros(1))
                    avg_downstream_cost_new = (1 - baseline_beta) * downstream_cost + \
                        baseline_beta * avg_downstream_cost_old
                    avg_downstream_cost_old.data = avg_downstream_cost_new.data  # XXX copy_() ?
                    baseline += avg_downstream_cost_old
                if use_nn_baseline:
                    # block nn_baseline_input gradients except in baseline loss
                    baseline += nn_baseline(detach_iterable(nn_baseline_input))
                    nn_params = nn_baseline.parameters()
                    baseline_loss = torch.pow(Variable(downstream_cost.data) - baseline, 2.0).sum()
                    baseline_losses.append((baseline_loss, nn_params))
                elif use_baseline_value:
                    # it's on the user to make sure baseline_value tape only points to baseline params
                    baseline += baseline_value
                    nn_params = baseline_params
                    baseline_loss = torch.pow(Variable(downstream_cost.data) - baseline, 2.0).sum()
                    baseline_losses.append((baseline_loss, nn_params))
                if use_nn_baseline or use_decaying_avg_baseline or use_baseline_value:
                    elbo_reinforce_terms += (guide_trace[node][log_pdf_key] *
                                             (Variable(downstream_cost.data - baseline.data))).sum()
                else:
                    elbo_reinforce_terms += (guide_trace[node][log_pdf_key] *
                                             Variable(downstream_cost.data)).sum()

            # minimize losses for any neural network baselines
            aggregate_baseline_loss = 0.0
            aggregate_baseline_params = set()
            for loss, params in baseline_losses:
                aggregate_baseline_loss += loss
                aggregate_baseline_params.update(params)

            if len(aggregate_baseline_params) > 0:
                aggregate_baseline_loss.backward()
                if self.baseline_optim_step_fct is None:
                    self.optim_step_fct(aggregate_baseline_params)
                else:
                    # use baseline_optim_step_fct if user provided
                    self.baseline_optim_step_fct(aggregate_baseline_params)
                pyro.util.zero_grads(params)

        # the gradient of the surrogate loss yields our gradient estimator for the elbo
        surrogate_loss = - elbo_no_zero_expectation_terms - elbo_reinforce_terms

        # accumulate trainable parameters for gradient step
        # XXX should this also be reflected in the construction of the surrogate_loss instead?
        all_trainable_params = []
        # get trace params from model run
        if not self.model_fixed:
            for site in model_trace.keys():
                if model_trace[site]["type"] == "param":
                    all_trainable_params.append(model_trace[site]["value"])
        # get trace params from guide run
        if not self.guide_fixed:
            for site in guide_trace.keys():
                if guide_trace[site]["type"] == "param":
                    all_trainable_params.append(guide_trace[site]["value"])
        all_trainable_params = list(set(all_trainable_params))

        # compute gradients
        surrogate_loss.backward()
        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        pyro.util.zero_grads(all_trainable_params)

        return elbo.data[0]
