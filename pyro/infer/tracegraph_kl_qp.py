import torch
from torch.autograd import Variable
from collections import OrderedDict
import pyro
import pyro.poutine as poutine
import sys
from pyro.util import ng_zeros, detach_iterable


class TraceGraph_KL_QP(object):
    """
    A TraceGraph and Poutine-based implementation of SVI
    The gradient estimator is constructed along the lines of
    'Gradient Estimation Using Stochastic Computation Graphs'
    John Schulman, Nicolas Heess, Theophane Weber, Pieter Abbeel
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
        self.sites = None

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
        guide_trgraph = poutine.tracegraph(self.guide)(*args, **kwargs)
        guide_trace = guide_trgraph.get_trace()

        model_trgraph = poutine.tracegraph(poutine.replay(self.model, guide_trace))(*args, **kwargs)
        model_trace = model_trgraph.get_trace()

        # have the trace compute all the individual log pdf terms
        # so that they are available below
        guide_trace.log_pdf(), model_trace.log_pdf()

        # prepare a list of all the cost nodes, each of which is +- log_pdf
        # the parent information will be used below for rao-blackwellization
        # also flag if term can be removed because its gradient has zero expectation
        # XXX had to change get_parents -> get_ancestors below to accommodate coarser
        # graph structure correctly : (
        cost_nodes = []
        for name in model_trace.keys():
            mtn = model_trace[name]
            if mtn["type"] == "observe":
                cost_node = (mtn["log_pdf"], model_trgraph.get_ancestors(name, with_self=False), True)
                cost_nodes.append(cost_node)
            elif mtn["type"] == "sample":
                gtn = guide_trace[name]
                cost_node1 = (mtn["log_pdf"], model_trgraph.get_ancestors(name, with_self=True), True)
                zero_expectation = name in guide_trgraph.get_nonreparam_stochastic_nodes()
                cost_node2 = (- gtn["log_pdf"], guide_trgraph.get_ancestors(name, with_self=True),
                              not zero_expectation)
                cost_nodes.extend([cost_node1, cost_node2])

        elbo = 0.0
        elbo_no_zero_expectation_terms = 0.0
        elbo_reinforce_terms = 0.0

        # compute the elbo; if all stochastic nodes are reparameterizable, we're done
        # this bit is never differentiated: it's here for getting an estimate of the elbo itself
        for cost_node in cost_nodes:
            elbo += cost_node[0]

        # compute the elbo, removing terms whose gradient is zero
        # this is the bit that's actually differentiated
        for cost_node in cost_nodes:
            if cost_node[2]:
                elbo_no_zero_expectation_terms += cost_node[0]

        # if there are any non-reparameterized stochastic nodes,
        # include all the reinforce-like terms.
        # we include only downstream costs to reduce variance
        # optionally include decaying average baseline to further reduce variance
        # XXX should the average baseline be in the param store as below?
        baseline_mses = []
        for node in guide_trgraph.get_nonreparam_stochastic_nodes():
            use_decaying_avg_baseline = guide_trace[node]['fn'].use_decaying_avg_baseline
            if use_decaying_avg_baseline:
                avg_downstream_cost_old = pyro.param("__baseline_avg_downstream_cost_" + node, ng_zeros(1))
                baseline_beta = guide_trace[node]['fn'].baseline_beta
            downstream_cost = 0.0
            downstream_cost_non_zero = False
            node_descendants = guide_trgraph.get_descendants(node, with_self=True)
            for cost_node in cost_nodes:
                if any([p in node_descendants for p in cost_node[1]]):
                    downstream_cost += cost_node[0]
                    downstream_cost_non_zero = True
            if use_decaying_avg_baseline:
                avg_downstream_cost_new = (1 - baseline_beta) * downstream_cost +\
                    baseline_beta * avg_downstream_cost_old
                avg_downstream_cost_old.data = avg_downstream_cost_new.data  # XXX copy_() ?
            if downstream_cost_non_zero:  # XXX is this actually necessary?
                use_dependent_baseline = guide_trace[node]['fn'].baseline is not None
                baseline = 0.0
                if use_decaying_avg_baseline:
                    baseline += avg_downstream_cost_old
                    detached_baseline = baseline
                if use_dependent_baseline:
                    baseline_input = pyro.util.detach_iterable(guide_trace[node]['baseline_input'])
                    baseline += guide_trace[node]['fn'].baseline(baseline_input)
                    detached_baseline = baseline.detach()
                if use_dependent_baseline or use_decaying_avg_baseline:
                    elbo_reinforce_terms += guide_trace[node]['log_pdf'] * \
                        (Variable(downstream_cost.data) - baseline.detach())
                    if use_dependent_baseline:
                        nn_params = guide_trace[node]['fn'].baseline.parameters()
                        baseline_mse = torch.pow(Variable(downstream_cost.data) - baseline, 2.0)
                        baseline_mses.append((baseline_mse, nn_params))
                else:
                    elbo_reinforce_terms += guide_trace[node]['log_pdf'] * Variable(downstream_cost.data)

        # minimize losses for any neural network baselines
        for loss, params in baseline_mses:
            loss.backward()
            if self.baseline_optim_step_fct is None:
                self.optim_step_fct(params)
            else:
                self.baseline_optim_step_fct(params)
            pyro.util.zero_grads(params)

        # the gradient of the surrogate loss yields our gradient estimator for the elbo
        surrogate_loss = - elbo_no_zero_expectation_terms - elbo_reinforce_terms

        # accumulate trainable parameters for gradient step
        # XXX should this also be reflected in the construction of the surrogate_loss instead?
        all_trainable_params = []
        # get trace params from model run
        if not self.model_fixed:
            for name in model_trace.keys():
                if model_trace[name]["type"] == "param":
                    all_trainable_params.append(model_trace[name]["value"])
        # get trace params from guide run
        if not self.guide_fixed:
            for name in guide_trace.keys():
                if guide_trace[name]["type"] == "param":
                    all_trainable_params.append(guide_trace[name]["value"])
        all_trainable_params = list(set(all_trainable_params))

        # compute gradients
        surrogate_loss.backward()
        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        pyro.util.zero_grads(all_trainable_params)

        return elbo.data[0]
