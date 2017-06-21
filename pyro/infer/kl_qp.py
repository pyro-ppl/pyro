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
        self.batch_index = 0  # TODO by name for multiple map_datas

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
        self.score_multiplier = pyro.ones(1)

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
        self.trace[name]["sample"] = v_sample
        self.trace[name]["logpdf"] = log_q * self.score_multiplier.expand_as(log_q)
        self.trace[name]["reparam"] = dist.reparametrized

        return v_sample

    def _pyro_map_data(self, data, fn, name="", batch_size=0):
        # TODO: multiple map_datas will require naming.

        if isinstance(data, torch.Tensor):
            # assume vectorized observation fn
            raise NotImplementedError(
                "map_data for vectorized data not yet implemented.")
        else:
            if batch_size == 0:
                batch_size = len(data)

            batch_ratio = len(data) / batch_size

            # get a minibatch
            # FIXME: this is brittle, deal with mismatched stride.
            if self.batch_index >= len(data):
                self.batch_index = 0
            new_batch_index = self.batch_index + batch_size

            # FIXME: should add batch as a trace node
            self.batch = list(enumerate(data))[
                self.batch_index:new_batch_index]
            self.batch_index = new_batch_index

            # set multiplyer for logpdfs based on batch size, used wheres score
            # are made.
            self.score_multiplier = self.score_multiplier * batch_ratio

            # map over the minibatch
            # note that fn should expect an index and a datum
            map(lambda x: fn(x[0], x[1]), self.batch)

            # undo multiplyer
            self.score_multiplier = self.score_multiplier / batch_ratio


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

    def _pyro_map_data(self, data, fn, name="", batch_size=0):
        # TODO: multiple map_datas will require naming.

        if isinstance(data, torch.Tensor):
            # assume vectorized observation fn
            raise NotImplementedError(
                "map_data for vectorized data not yet implemented.")
        else:
            # use minibatch from guide
            assert(len(self.batch) > 0)

            # batch_size comes from batch pre-constructed in guide.
            batch_ratio = len(data) / len(self.batch)

            # set multiplyer for logpdfs based on batch size, used wheres score
            # are made.
            self.score_multiplier = self.score_multiplier * batch_ratio

            # map over the minibatch
            # note that fn should expect an index and a datum
            map(lambda x: fn(x[0], x[1]), self.batch)

            # undo multiplyer
            self.score_multiplier = self.score_multiplier / batch_ratio


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
            log_r += self.model.trace[name]["logpdf"] - \
                self.guide.trace[name]["logpdf"]
        for name in self.guide.trace.keys():
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
