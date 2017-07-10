import six
import pdb
import torch
from torch.autograd import Variable
from collections import OrderedDict
import pyro
import pyro.poutine as poutine
from pyro.infer.abstract_infer import AbstractInfer


def zero_grads(tensors):
    """
    Sets gradients of list of Variables to zero in place
    """
    for p in tensors:
        if p.grad is not None:
            if p.grad.volatile:
                p.grad.data.zero_()
            else:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())


class TraceKLqp(AbstractInfer):
    """
    A new, Trace and Poutine-based implementation of SVI
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
        super(TraceKLqp, self).__init__()
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
        guide_trace = poutine.trace(self.guide)(*args, **kwargs)
        model_trace = poutine.trace(
            poutine.replay(self.model, guide_trace))(*args, **kwargs)

        # compute losses
        log_r = model_trace.log_pdf() - guide_trace.log_pdf()

        elbo = 0.0
        for name in model_trace.keys():
            if model_trace[name]["type"] == "observe":
                elbo += model_trace[name]["log_pdf"]
            elif model_trace[name]["type"] == "sample":
                if model_trace[name]["fn"].reparametrized:
                    elbo += model_trace[name]["log_pdf"]
                    elbo -= guide_trace[name]["log_pdf"]
                else:
                    elbo += Variable(log_r.data) * guide_trace[name]["log_pdf"]
            else:
                pass

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
        loss = -elbo
        loss.backward()
        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        zero_grads(all_trainable_params)

        return loss.data[0]
