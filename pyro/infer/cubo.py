import six
import torch
from torch.autograd import Variable
from collections import OrderedDict
import pyro
import pyro.poutine as poutine
from pyro.infer.abstract_infer import AbstractInfer

import pdb as pdb

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


class CUBO(AbstractInfer):
    """
    A new, Trace and Poutine-based implementation of SVI
    """
    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False,
                 n_cubo=2,
                 nr_particles = 3,
                 *args, **kwargs):
        """
        Call parent class initially, then setup the poutines to run
        """
        # initialize
        super(CUBO, self).__init__()
        # TODO init this somewhere else in a more principled way
        self.sites = None

        self.model = model
        self.guide = guide
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed
        self.n_cubo = n_cubo
        self.nr_particles = nr_particles

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        single step?
        """
        model_traces = []
        guide_traces = []
        log_weights = []
        for i in range(self.nr_particles):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)

            log_r_raw = model_trace.batch_log_pdf() - guide_trace.batch_log_pdf()
            log_weights.append(log_r_raw)
            model_traces.append(model_trace)
            guide_traces.append(guide_trace)


        log_weights_tensor = torch.stack(log_weights, 1)

        # max samples along first dimenaion to get c
        log_r_max = torch.max(log_weights_tensor, 1)[0]

        # w(s)
        log_r = log_weights_tensor - log_r_max.expand_as(log_weights_tensor)

        w_n = Variable(torch.exp(log_r * self.n_cubo).data)

        cubo = 0.0
        exp_cubo = 0.0
        grad_cubo = 0.0
        for i in range(self.nr_particles):
            log_r_s = 0.0
            model_trace = model_traces[i]
            guide_trace = guide_traces[i]
            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    log_r_s += model_trace[name]["batch_log_pdf"] * (self.n_cubo) / self.nr_particles

                elif model_trace[name]["type"] == "sample":
                    if model_trace[name]["fn"].reparametrized:
                        # print "name",model_trace[name]
                        log_r_s += w_n[:,i] * model_trace[name]["batch_log_pdf"] * (self.n_cubo) / self.nr_particles
                        log_r_s -= w_n[:,i] * guide_trace[name]["batch_log_pdf"] * (self.n_cubo) / self.nr_particles

                    else:
                        log_r_s += w_n[:,i] * guide_trace[name]["batch_log_pdf"] * (1-self.n_cubo) / self.nr_particles
                else:
                    pass

            exp_cubo += torch.exp(log_r_s * self.n_cubo) / self.nr_particles
            grad_cubo += log_r_s #/self.nr_particles

        # exp_cubo_sum = grad_cubo.sum()
        # use estimator
        exp_cubo_sum = exp_cubo.sum()

        cubo = (torch.log(exp_cubo)/self.n_cubo ).sum()

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
        loss = exp_cubo_sum
        loss.backward()
        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        zero_grads(all_trainable_params)

        # return the log transform of the expectation
        return cubo.data[0]
