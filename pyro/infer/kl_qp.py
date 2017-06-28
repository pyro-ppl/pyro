import pyro
from pyro.infer.poutine import TagPoutine
from pyro.infer.abstract_infer import AbstractInfer
import torch
from torch.autograd import Variable
from collections import OrderedDict


class VIGuideCo(TagPoutine):
    """
    The guide copoutine should:
    1) cache a sample at each site
    2) compute and store log(q(sample)) at each site
    3) log each site's distribution type or at least record reparametrizable
    """
    # uniquely tag each trace -- not expensive

    def tag_name(self, trace_uid):
        return "guide_{}".format(trace_uid)

    def __init__(self, *args, **kwargs):
        super(VIGuideCo, self).__init__(*args, **kwargs)

    # every time
    def _enter_poutine(self, *args, **kwargs):
        """
        When model execution begins
        """
        super(VIGuideCo, self)._enter_poutine(*args, **kwargs)

        # trace structure:
        # site = {"sample": sample, "logq": logq, "reparam": reparam}
        self.trace = OrderedDict()
        self.batch = []
        # TODO: make this cleaner via a trace data structure
        self.score_multiplier = Variable(torch.ones(1))

    def _pyro_sample(self, name, dist):
        """
        Simply sample from distribution
        """

        if dist.reparametrized:
            v_sample = dist()
        else:
            # Do we need to untape here, or will it already be untaped because the
            #  sampler has a random choice (after params)?
            v_sample = Variable(dist().data)  # XXX should be detach?
        log_q = dist.batch_log_pdf(v_sample)

        assert(name not in self.trace)
        self.trace[name] = {}
        self.trace[name]["type"] = "sample"
        self.trace[name]["sample"] = v_sample
        self.trace[name]["logpdf"] = log_q * self.score_multiplier.expand_as(log_q)
        self.trace[name]["reparam"] = dist.reparametrized

        return v_sample

    def _pyro_map_data(self, name, fn, data, batch_size=0):
        """
        Guide Poutine map_data implementation.
        In this version, select a random subset of data and map over that
        Then record the indices of the minibatch in the trace.
        """
        assert(name not in self.trace)
        
        if isinstance(data, torch.Tensor) or isinstance(data, Variable):
            # draw a batch
            if batch_size == 0:
                batch_size = data.size(0)
            ind = torch.randperm(data.size(0))[0:batch_size]
            # create the trace node
            self.trace[name] = {}
            self.trace[name]["type"] = "map_data"
            self.trace[name]["indices"] = ind
            batch_ratio = float(data.size(0)) / float(batch_size)
            self.score_multiplier = self.score_multiplier * batch_ratio
            vals = fn(ind, data.index(0, ind))
            self.score_multiplier = self.score_multiplier / batch_ratio
            return vals
        else:
            # data is a non-tensor iterable
            if batch_size == 0:
                batch_size = len(data)

            ind = list(torch.randperm(len(data))[0:batch_size])

            self.trace[name] = {}
            self.trace[name]["type"] = "map_data"
            self.trace[name]["indices"] = ind
            
            batch_ratio = float(len(data)) / float(batch_size)
            self.score_multiplier = self.score_multiplier * batch_ratio
            vals = list(map(lambda ix: fn(*ix), zip(ind, [data[i] for i in ind])))
            self.score_multiplier = self.score_multiplier / batch_ratio
            return vals


class VIModelCo(TagPoutine):

    # uniquely tag each trace -- not expensive
    def tag_name(self, trace_uid):
        return "model_{}".format(trace_uid)

    def set_trace(self, guide_trace):
        self.guide_trace = guide_trace

        # fixme: should put batch in trace instead
    def set_batch(self, guide_batch):
        self.batch = guide_batch

    # every time
    def _enter_poutine(self, *args, **kwargs):
        """
        When model execution begins
        """
        super(VIModelCo, self)._enter_poutine(*args, **kwargs)
        self.observation_LL = 0
        self.score_multiplier = 1
        self.trace = OrderedDict()

    def _pyro_sample(self, name, dist):
        """
        Simply sample from distribution
        """
        v_sample = self.guide_trace[name]["sample"]
        # typecheck class?
        log_p = dist.batch_log_pdf(v_sample)

        assert(name not in self.trace)

        self.trace[name] = {}
        self.trace[name]["type"] = "sample"
        self.trace[name]["logpdf"] = log_p * self.score_multiplier
        self.trace[name]["sample"] = v_sample
        self.trace[name]["reparam"] = dist.reparametrized

        return v_sample

    def _pyro_observe(self, name, dist, val):
        """
        Get log_pdf of sample, add to ongoing scoring
        """
        logpdf = dist.batch_log_pdf(val)
        self.observation_LL = self.observation_LL + logpdf * self.score_multiplier
        return val

    def _pyro_map_data(self, name, fn, data, batch_size=0):
        """
        Model Poutine map_data.
        In this version, we just reuse the minibatch indices from the guide
        instead of using a fresh random sample.
        Otherwise identical to the guide's version.
        """
        assert(name not in self.trace and name in self.guide_trace)
        if isinstance(data, torch.Tensor) or isinstance(data, Variable):
            # draw a batch
            if batch_size == 0:
                batch_size = data.size(0)
            ind = self.guide_trace[name]["indices"]
            # create the trace node
            self.trace[name] = {}
            self.trace[name]["type"] = "map_data"
            self.trace[name]["indices"] = ind
            batch_ratio = float(data.size(0)) / float(batch_size)
            self.score_multiplier = self.score_multiplier * batch_ratio
            vals = fn(ind, data.index(0, ind))
            self.score_multiplier = self.score_multiplier / batch_ratio
            return vals
        else:
            # data is a non-tensor iterable
            if batch_size == 0:
                batch_size = len(data)

            ind = self.guide_trace[name]["indices"]

            self.trace[name] = {}
            self.trace[name]["type"] = "map_data"
            self.trace[name]["indices"] = ind
            
            batch_ratio = float(len(data)) / float(batch_size)
            self.score_multiplier = self.score_multiplier * batch_ratio
            vals = list(map(lambda ix: fn(*ix), zip(ind, [data[i] for i in ind])))
            self.score_multiplier = self.score_multiplier / batch_ratio
            return vals


class KL_QP(AbstractInfer):

    def __init__(self, model,
                 guide,
                 optim_step_fct,
                 model_fixed=False,
                 guide_fixed=False, *args, **kwargs):
        """
        Call parent class initially, then setup the copoutines to run
        """
        # initialize
        super(KL_QP, self).__init__()

        # wrap the model function with a LWCoupoutine
        self.model = VIModelCo(model)
        self.guide = VIGuideCo(guide)
        self.optim_step_fct = optim_step_fct
        self.model_fixed = model_fixed
        self.guide_fixed = guide_fixed

    def __call__(self, *args, **kwargs):
        return self.step(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        Main function of an Infer object, automatically switches context with copoutine
        """
        # for each step, sample guide, sample model,
        # collect all the relevant parameters for both the guide
        # and model according to whether or not those parameters
        # will be updated by optim
        all_trainable_params = []

        # sample from the guide
        # this will store random variables in self.guide.trace
        self.guide(*args, **kwargs)

        # get trace params from last guide run
        # i.e. all the calls to pyro.param inside of guide
        if not self.guide_fixed:
            all_trainable_params += self.guide.get_last_trace_parameters()

        # use guide trace inside of our model copoutine
        self.model.set_trace(self.guide.trace)
        self.model.set_batch(self.guide.batch)

        # sample from model, using the guide trace
        self.model(*args, **kwargs)

        # get trace params from last model run
        # i.e. all the calls to pyro.param inside of model
        if not self.model_fixed:
            all_trainable_params += self.model.get_last_trace_parameters()

        # now we finish computing elbo
        # loop over non-reparametrized names
        # current score reflects observations during trace
        log_r = self.model.observation_LL.clone()
        elbo = self.model.observation_LL

        for name in self.guide.trace.keys():
            if self.guide.trace[name]["type"] == "sample":
                log_r += self.model.trace[name]["logpdf"] - \
                    self.guide.trace[name]["logpdf"]
        for name in self.guide.trace.keys():
            if self.guide.trace[name]["type"] == "sample":
                if not self.guide.trace[name]["reparam"]:
                    elbo += Variable(log_r.data) * self.guide.trace[name]["logpdf"]
                else:
                    elbo += self.model.trace[name]["logpdf"]
                    elbo -= self.guide.trace[name]["logpdf"]

        loss = -torch.sum(elbo)
        # prxint("model loss {}".format(loss.data[0]))
        # use loss variable to calculate backward
        # thanks pytorch autograd
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
