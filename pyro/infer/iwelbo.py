from pyro.infer.abstract_infer import AbstractInfer
from pyro.infer.kl_qp.py import VIGuideCo
from pyro.infer.kl_qp.py import VIModelCo
import torch
from torch.autograd import Variable


class IWELBo(AbstractInfer):

    def __init__(self, model,
                 guide,
                 optim_fct,
                 model_fixed=False,
                 guide_fixed=False, *args, **kwargs):
        """
        Call parent class initially, then setup the copoutines to run
        """
        # initialize
        super(IWELBo, self).__init__()

        # wrap the model function with a CoPoutine
        # this will push and pop monad as needed
        self.model = VIModelCo(model)
        self.guide = VIGuideCo(guide)
        self.optim_fct = optim_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed

    def __call__(self, *args, **kwargs):
        return self.runner(*args, **kwargs)

    def runner(self, *args, **kwargs):
        """
        Main function of an Infer object, automatically switches context with copoutine
        """
        num_steps = 1
        num_importance_samples = 2
        # for each step, sample guide, sample model,
        for i in range(num_steps):
            elbo = 0.0  # FIXME: need to make variable?
            ws = []  # TODO vectorize importance samples?
            all_trainable_params = []

            for j in range(num_importance_samples):

                # sample from the guide
                # this will store random variables in self.guide.trace
                self.guide(*args, **kwargs)

                # get trace params from last guide run
                # i.e. all the calls to pyro.param inside of guide
                if not self.guide_fixed:
                    all_trainable_params += self.guide.get_last_trace_parameters()

                # use guide trace inside of our model copoutine
                self.model.set_trace(self.guide.trace)

                # sample from model, using the guide trace
                self.model(*args, **kwargs)

                # get trace params from last model run
                # i.e. all the calls to pyro.param inside of model
                if not self.model_fixed:
                    all_trainable_params += self.model.get_last_trace_parameters()

                log_r = self.model.observation_LL
                elbo += self.model.observation_LL
                for name in self.guide.trace.keys():
                    log_r += self.model.trace[name]["logpdf"] - \
                        self.guide.trace[name]["logpdf"]
                for name in self.guide.trace.keys():
                    if not self.guide.trace[name]["reparam"]:
                        # FIXME: i'm not sure if iawe math works for LR
                        # estimator.
                        raise NotImplementedError(
                            "need to work out LR case for importance-weighted objective")
                        # elbo += Variable(log_r.data) * self.guide.trace[name]["logpdf"]
                    else:
                        w = torch.exp(Variable(log_r.data))
                        ws += w
                        elbo += w * \
                            (self.model.trace[name]["logpdf"] -
                             self.guide.trace[name]["logpdf"])

            elbo /= sum(ws)  # need torch.sum?
            loss = -elbo
            loss.backward()

            # make sure we're only listing the unique trainable params
            all_trainable_params = list(set(all_trainable_params))

            # construct our optim object EVERY step
            # TODO: Make this more efficient with respect to params
            self.optim_fct(all_trainable_params, lr=.000001).step()
            """Sets gradients of all model parameters to zero."""
            for p in all_trainable_params:
                if p.grad is not None:
                    if p.grad.volatile:
                        p.grad.data.zero_()
                    else:
                        data = p.grad.data
                        p.grad = Variable(data.new().resize_as_(data).zero_())
