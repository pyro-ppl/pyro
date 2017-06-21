import six
import torch
from torch.autograd import Variable
from collections import OrderedDict
import pyro
from pyro.infer.trace import TracePoutine
from pyro.infer.abstract_infer import AbstractInfer

class TraceKLqp(AbstractInfer):
    """hello"""

    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False, *args, **kwargs):
        """
        Call parent class initially, then setup the copoutines to run
        """
        # initialize
        super(TraceKLqp, self).__init__()
        # TODO init this somewhere else in a more principled way
        self.sites = None

        # wrap the model function with a TracePoutine
        self.model = TracePoutine(model)
        self.guide = TracePoutine(guide)
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        single elbo optimization step
        """
        # compute
        self.model.flush_traces()
        self.guide.flush_traces()

        all_trainable_params = []

        # sample from the guide
        #bb()
        guide_ret = self.guide(*args, **kwargs)

        # get trace params from last guide run
        if not self.guide_fixed:
            all_trainable_params += self.guide.get_last_trace_parameters()

        # sample from model, using the guide trace
        rv = self.model.replay(
            self.guide.trace, sites=self.sites, *args, **kwargs)

        # get trace params from last model run
        if not self.model_fixed:
            all_trainable_params += self.model.get_last_trace_parameters()

        # get obs LL
        logp_obs = Variable(torch.zeros(1))
        for name in self.model.trace:
            if self.model.trace[name]["type"] == "observe":
                obs_args, obs_kwargs = self.model.trace[name]["args"]
                logp_obs += self.model.trace[name]["dist"].log_pdf(
                    self.model.trace[name]["value"], *obs_args, **obs_kwargs)

        # current score reflects observations during trace
        log_r = logp_obs.clone()
        elbo = logp_obs.clone()

        # XXX should cache pdf computations
        for name in self.guide.trace.keys():
            if self.guide.trace[name]["type"] == "sample":
                # log(p(z))
                model_site_args, model_site_kwargs = self.model.trace[name]["args"]
                log_r += self.model.trace[name]["dist"].log_pdf(
                    self.model.trace[name]["value"],
                    *model_site_args, **model_site_kwargs)
                # log(q(z))
                guide_site_args, guide_site_kwargs = self.guide.trace[name]["args"]
                log_r -= self.guide.trace[name]["dist"].log_pdf(
                    self.guide.trace[name]["value"],
                    *guide_site_args, **guide_site_kwargs)

        # XXX distinction between reparam and not is artificial, leads to repetition
        for name in self.guide.trace.keys():
            if self.guide.trace[name]["type"] == "sample":
                if not self.guide.trace[name]["dist"].reparametrized:
                    guide_site_args, guide_site_kwargs = self.guide.trace[name]["args"]
                    # XXX should be doing this via reinforce()
                    elbo += Variable(log_r.data) * self.guide.trace[name]["dist"].log_pdf(
                        self.guide.trace[name]["value"],
                        *guide_site_args, **guide_site_kwargs)
                else:
                    # log(p(z))
                    model_site_args, model_site_kwargs = self.model.trace[name]["args"]
                    elbo += self.model.trace[name]["dist"].log_pdf(
                        self.model.trace[name]["value"],
                        *model_site_args, **model_site_kwargs)
                    # log(q(z))
                    guide_site_args, guide_site_kwargs = self.guide.trace[name]["args"]
                    elbo -= self.guide.trace[name]["dist"].log_pdf(
                        self.guide.trace[name]["value"],
                        *guide_site_args, **guide_site_kwargs)

        loss = -elbo
        loss.backward()

        # make sure we're only listing the unique trainable params
        all_trainable_params = list(set(all_trainable_params))

        # construct our optim object EVERY step
        # TODO: Make this more efficient with respect to params
        self.optim_step_fct(all_trainable_params)
        """Sets gradients of all model parameters to zero."""
        for p in all_trainable_params:
            if p.grad is not None:
                if p.grad.volatile:
                    p.grad.data.zero_()
                else:
                    data = p.grad.data
                    p.grad = Variable(data.new().resize_as_(data).zero_())

        # send back the float loss plz
        return loss.data[0]
