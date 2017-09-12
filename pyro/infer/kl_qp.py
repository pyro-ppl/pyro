import torch
from torch.autograd import Variable
from collections import OrderedDict
import pyro
import pyro.poutine as poutine
# from pyro.infer.abstract_infer import AbstractInfer
import pdb as pdb


class KL_QP(object):  # AbstractInfer):
    """
    :param model: probabilistic model defined as a function
    :param guide: guide used for sampling defined as a function
    :param optim: optimization function
    :param model_fixed: flag for if the model is fixed
    :type model_fixed: bool
    :param guide_fixed: flag for if the guide is fixed
    :type guide_fixed: bool

    This method performs variational inference by minimizing the
    `KL-divergence <https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence>`_
    between the actual and approximate posterior.

    Example::

        from pyro.infer.kl_qp import KL_QP

        kl_optim = KL_QP(model, guide,
                         pyro.optim(torch.optim.Adam, {"lr": .001}))
        for k in range(n_steps):
        # optimize
        kl_optim.step()
    """
    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False,
                 nr_particles = 1,
                 *args, **kwargs):
        """
        Call parent class initially, then setup the poutines to run
        """
        # initialize
        super(KL_QP, self).__init__()
        # TODO init this somewhere else in a more principled way
        self.sites = None
        self.model = model
        self.guide = guide
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed
        self.nr_particles = nr_particles

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)


    def eval_bound(self, *args, **kwargs):
        """
        Evaluate Elbo by running nr_particles often.
        This nr_particles can differ from the one used globally for gradient estimation if set, but generally matches global values.
        Returns the Elbo as a value
        """
        model_traces = []
        guide_traces = []
        log_r_per_sample = []

        nr_particles = self.nr_particles

        for i in range(nr_particles):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)

            log_r = model_trace.log_pdf() - guide_trace.log_pdf()
            log_r_per_sample.append(log_r)
            model_traces.append(model_trace)
            guide_traces.append(guide_trace)


        elbo = 0.0
        for i in range(nr_particles):
            elbo_particle = 0.0
            log_r_s = 0.0
            model_trace = model_traces[i]
            guide_trace = guide_traces[i]
            
            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    elbo_particle += model_trace[name]["log_pdf"]
                elif model_trace[name]["type"] == "sample":
                    elbo_particle += model_trace[name]["log_pdf"]
                    elbo_particle -= guide_trace[name]["log_pdf"]

                else:
                    pass
            elbo += elbo_particle
       

        # gradients
        loss = -elbo

        return loss.data[0]


    def eval_grad(self, *args, **kwargs):
        """
        Evaluates the statistics for a single gradient step of ELBO based on nr_particles many samples.
        """

        model_traces = []
        guide_traces = []
        log_r_per_sample = []

        nr_particles = self.nr_particles

        for i in range(nr_particles):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)

            log_r = model_trace.log_pdf() - guide_trace.log_pdf()
            log_r_per_sample.append(log_r)
            model_traces.append(model_trace)
            guide_traces.append(guide_trace)

        # guide_trace = poutine.trace(self.guide)(*args, **kwargs)
        # model_trace = poutine.trace(
        #     poutine.replay(self.model, guide_trace))(*args, **kwargs)

        # # compute losses
        # log_r = model_trace.log_pdf() - guide_trace.log_pdf()

        elbo = 0.0
        for i in range(nr_particles):
            elbo_particle = 0.0
            log_r_s = 0.0
            model_trace = model_traces[i]
            guide_trace = guide_traces[i]
            log_r = log_r_per_sample[i]

            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    elbo_particle += model_trace[name]["log_pdf"]
                elif model_trace[name]["type"] == "sample":
                    if model_trace[name]["fn"].reparameterized:
                        elbo_particle += model_trace[name]["log_pdf"]
                        elbo_particle -= guide_trace[name]["log_pdf"]
                    else:
                        elbo_particle += Variable(log_r.data) * guide_trace[name]["log_pdf"]
                else:
                    pass
            elbo += elbo_particle

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

        return loss, all_trainable_params


    def step(self, *args, **kwargs): #nr_particles=None, loss=None,
        """
        Takes a single step of optimization by using the elbo loss and then applying autograd
        Returns the Elbo as a value

        If "loss" has been precomputed it can be passed into it, else it runs eval_grad with nr_partricles
        """

        nr_particles = self.nr_particles

        if 'loss' not in kwargs.keys():
            
            [loss, all_trainable_params] = self.eval_grad(*args, **kwargs)
        
        else:
            [loss,all_trainable_params] = kwargs['loss']

        loss.backward()

        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        pyro.util.zero_grads(all_trainable_params)

        return loss.data[0]
