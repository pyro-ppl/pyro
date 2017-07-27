import torch
from torch.autograd import Variable
from collections import OrderedDict
import pyro
import pyro.poutine as poutine

class TraceGraph_KL_QP(object):
    """
    A Tracegraph and Poutine-based implementation of SVI
    """
    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False, *args, **kwargs):
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

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        single step?
        """
        step_number = kwargs.pop('step_number', -1)
        reparam = kwargs.pop('reparameterized', '')
        if step_number==0:
            guide_tracegraph = poutine.tracegraph(self.guide, graph_output='guide.%s' % reparam)(*args, **kwargs)
        else:
            guide_tracegraph = poutine.tracegraph(self.guide)(*args, **kwargs)
        guide_trace = guide_tracegraph.get_trace()
        model_tracegraph = poutine.tracegraph(
             poutine.replay(self.model, guide_trace))(*args, **kwargs)
        model_trace = model_tracegraph.get_trace()

        # compute losses
        _ = guide_trace.log_pdf() - model_trace.log_pdf()

        surrogate_loss_stochastic_bit = 0.0
        surrogate_loss_deterministic_bit = 0.0

        for name in model_trace.keys():
            if model_trace[name]["type"] == "observe":
                surrogate_loss_deterministic_bit -= model_trace[name]["log_pdf"]
            elif model_trace[name]["type"] == "sample":
                surrogate_loss_deterministic_bit -= model_trace[name]["log_pdf"]
                surrogate_loss_deterministic_bit += guide_trace[name]["log_pdf"]

        for node in guide_tracegraph.get_direct_stochastic_children_of_parameters():
            for cost_node in model_trace.keys():
                if cost_node in guide_tracegraph.get_descendants(node, with_self=True):
                    if model_trace[cost_node]["type"] == "sample":
                        surrogate_loss_stochastic_bit -= Variable(model_trace[cost_node]["log_pdf"].data) *\
                                                         guide_trace[node]['log_pdf']
                        surrogate_loss_stochastic_bit += Variable(guide_trace[cost_node]["log_pdf"].data) *\
                                                         guide_trace[node]['log_pdf']
                if model_trace[cost_node]["type"] == "observe":
                    include_term = False
                    for parent in model_tracegraph.observe_stochastic_parents[cost_node]:
                        if parent in guide_tracegraph.get_ancestors(parent, with_self = True):
                            include_term = True
                    if include_term:
                        surrogate_loss_stochastic_bit -= Variable(model_trace[cost_node]["log_pdf"].data) *\
                                                         guide_trace[node]['log_pdf']

        surrogate_loss = surrogate_loss_stochastic_bit + surrogate_loss_deterministic_bit
        elbo = -surrogate_loss_deterministic_bit

        # accumulate parameters
        all_trainable_params = []
        # get trace params from last model run
        if not self.model_fixed:
            for name in model_trace.keys():
                if model_trace[name]["type"] == "param":
                    all_trainable_params.append(model_trace[name]["value"])
        # get trace params from last guide run
        if not self.guide_fixed:
            for name in guide_trace.keys():
                if guide_trace[name]["type"] == "param":
                    all_trainable_params.append(guide_trace[name]["value"])
        all_trainable_params = list(set(all_trainable_params))

        # gradients
        surrogate_loss.backward()
        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        pyro.util.zero_grads(all_trainable_params)

        return elbo.data[0]
