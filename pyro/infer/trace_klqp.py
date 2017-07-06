import six
import torch
from torch.autograd import Variable
from collections import OrderedDict
import pyro
import pyro.infer.poutine as poutine
from pyro.infer.trace import TracePoutine
from pyro.infer.abstract_infer import AbstractInfer

def zero_grads(all_trainable_params):
    """
    Sets gradients of all model parameters to zero
    """
    for p in all_trainable_params:
        if p.grad is not None:
            if p.grad.volatile:
                p.grad.data.zero_()
            else:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())


class TraceKLqp(AbstractInfer):
    """hello"""

    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False, *args, **kwargs):
        """
        Call parent class initially, then setup the poutines to run
        """
        # initialize
        super(TraceKLqp, self).__init__()
        # TODO init this somewhere else in a more principled way
        self.sites = None

        self.model = model #TracePoutine(model)
        self.guide = guide #TracePoutine(guide)
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)


    def step(self, *args, **kwargs):
        """
        single step?
        """
        guide_trace = poutine.trace(guide)(*args, **kwargs)
        model_trace = poutine.trace(
            poutine.replay(model, guide_trace))(*args, **kwargs)

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

        # compute losses
        # TODO get reparam right
        elbo = 0.0
        for name in model_trace.keys():
            if model_trace[name]["type"] == "observe":
                elbo = elbo + model_trace[name]["dist"].log_pdf(
                    model_trace[name]["value"],
                    *model_trace[name]["args"][0],
                    **model_trace[name]["args"][1])
            elif model_trace[name]["type"] == "sample":
                elbo = elbo + model_trace[name]["dist"].log_pdf(
                    model_trace[name]["value"],
                    *model_trace[name]["args"][0],
                    **model_trace[name]["args"][1])
                elbo = elbo - guide_trace[name]["dist"].log_pdf(
                    guide_trace[name]["value"],
                    *guide_trace[name]["args"][0],
                    **guide_trace[name]["args"][1])

        # gradients
        (-elbo).backward()
        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        zero_grads(all_trainable_params)

        return elbo.data[0]
    
