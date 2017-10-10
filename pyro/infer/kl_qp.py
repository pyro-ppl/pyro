import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine


class KL_QP(object):
    """
    :param model: probabilistic model defined as a function
    :param guide: guide used for sampling defined as a function
    :param optim: optimization function
    :param model_fixed: flag that controls whether the model parameters are fixed during optimization
    :type model_fixed: bool
    :param guide_fixed: flag that controls whether the guide parameters are fixed during optimization
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
                 num_particles=1,
                 enum_discrete=False,
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
        self.num_particles = num_particles
        self.enum_discrete = enum_discrete

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def iter_traces(self, *args, **kwargs):
        """
        Method to draw,store and evaluate multiple samples in traces
        """
        for i in range(self.num_particles):
            # THIS IS A TOTAL HACK
            if self.enum_discrete:
                from six.moves.queue import LifoQueue
                from pyro.poutine.trace import Trace

                def is_discrete(name, site):
                    return getattr(site, "enumerable", False)

                queue = LifoQueue()
                queue.put(Trace())
                next_guide = poutine.queue(poutine.trace(self.guide), queue)
                while not queue.empty():
                    guide_trace = next_guide(*args, **kwargs)
                    model_trace = poutine.trace(
                        poutine.replay(self.model, guide_trace))(*args, **kwargs)
                    log_r = model_trace.log_pdf() - guide_trace.log_pdf()
                    log_q_discrete = guide_trace.log_pdf(is_discrete)
                    weight = torch.exp(log_q_discrete.detach())  # Block gradients.
                    yield weight, model_trace, guide_trace, log_r
                continue

            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)
            log_r = model_trace.log_pdf() - guide_trace.log_pdf()

            yield 1.0, model_trace, guide_trace, log_r

    def eval_objective(self, *args, **kwargs):
        """
        Evaluate Elbo by running num_particles often.
        Returns the Elbo as a value
        """
        elbo = 0.0
        for weight, model_trace, guide_trace, log_r in self.iter_traces(*args, **kwargs):
            elbo_particle = 0.0

            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    elbo_particle += model_trace[name]["log_pdf"]
                elif model_trace[name]["type"] == "sample":
                    elbo_particle += model_trace[name]["log_pdf"]
                    elbo_particle -= guide_trace[name]["log_pdf"]
                else:
                    pass
            elbo += elbo_particle * (weight / self.num_particles)
        loss = -elbo

        return loss.data[0]

    def eval_grad(self, *args, **kwargs):
        """
        Computes a surrogate loss, which, when differentiated yields an estimate of the gradient of the Elbo.
        Num_particle many samples are used to form the surrogate loss.
        """
        elbo = 0.0
        all_trainable_params = []

        for weight, model_trace, guide_trace, log_r in self.iter_traces(*args, **kwargs):
            elbo_particle = 0.0

            for name in model_trace.keys():
                if model_trace[name]["type"] == "observe":
                    elbo_particle += model_trace[name]["log_pdf"]
                elif model_trace[name]["type"] == "sample":
                    if model_trace[name]["fn"].reparameterized:
                        elbo_particle += model_trace[name]["log_pdf"]
                        elbo_particle -= guide_trace[name]["log_pdf"]
                    else:
                        elbo_particle += Variable(log_r.data) * guide_trace[name]["log_pdf"] +\
                                         model_trace[name]["log_pdf"]
                else:
                    pass
            elbo += elbo_particle * (weight / self.num_particles)

            # accumulate parameters
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
        loss = -elbo

        return loss, all_trainable_params

    def step(self, *args, **kwargs):
        """
        Takes a single step of optimization by using the elbo loss and then applying autograd
        Returns the Elbo as a value
        If "loss" has been precomputed it can be passed into it, else it runs eval_grad with num_partricles
        """

        if 'loss_and_params' not in kwargs.keys():
            [loss, all_trainable_params] = self.eval_grad(*args, **kwargs)
        else:
            [loss, all_trainable_params] = kwargs['loss_and_params']

        loss.backward()

        # update
        self.optim_step_fct(all_trainable_params)
        # zero grads
        pyro.util.zero_grads(all_trainable_params)

        return loss.data[0]
