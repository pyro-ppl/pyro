from pyro.poutine import TagPoutine
from pyro.infer.abstract_infer import AbstractInfer
from torch.autograd import Variable
import torch


class GuideCo(TagPoutine):

    # uniquely tag each trace -- not expensive because
    # params are
    def tag_name(self, trace_uid):
        return "guide_{}".format(trace_uid)

    # every time
    def _enter_poutine(self, *args, **kwargs):
        """
        When model execution begins
        """
        super(GuideCo, self)._enter_poutine(*args, **kwargs)
        self.current_score = Variable(torch.zeros(1))  # 0.
        self.trace = {}

    def _pyro_sample(self, name, dist):
        """
        Simply sample from distribution
        """
        v = dist()
        s = dist.log_pdf(v)
        self.current_score += s
        self.trace[name] = v
        return v


class ModelCo(TagPoutine):

    def tag_name(self, trace_uid):
        """
        Tag every parameters with model_{trace_uid}.
        VI would call for all parameters called in the last trace.
        """
        return "model_{}".format(trace_uid)

    def set_trace(self, trace):
        self.guide_trace = trace

    # every time
    def _enter_poutine(self, *args, **kwargs):
        """
        When model execution begins
        """
        super(ModelCo, self)._enter_poutine(*args, **kwargs)
        self.current_score = Variable(torch.zeros(1))  # 0.

    def _pyro_sample(self, name, dist):
        """
        Simply sample from distribution
        """
        v = self.guide_trace[name]
        s = dist.log_pdf(v)
        self.current_score += s
        return v

    def _pyro_observe(self, name, dist, val):
        """
        Get log_pdf of sample, add to ongoing scoring
        """
        logp = dist.log_pdf(val)
        self.current_score += logp
        return val


class ImportanceSampling(AbstractInfer):

    def __init__(self, model, guide, *args, **kwargs):
        """
        Call parent class initially, then setup the couroutines to run
        """
        # initialize
        super(ImportanceSampling, self).__init__()

        # wrap the model function with a LWCoupoutine
        # this will push and pop state
        self.model = ModelCo(model)
        self.guide = GuideCo(guide)

    def runner(self, num_samples, *args, **kwargs):
        """
        Main function of an Infer object, automatically switches context with copoutine
        """
        # setup sample to hold
        samples = []

        for i in range(num_samples):
            # sample from the guide
            # this will store random variables in self.guide.trace
            self.guide(*args, **kwargs)

            # use guide trace inside of our model copoutine
            self.model.set_trace(self.guide.trace)

            # sample from model, using the guide trace
            rv = self.model(*args, **kwargs)

            # add to sample state
            samples.append(
                [i, rv, self.model.current_score - self.guide.current_score])

        # send back array of samples to be consumed elsewhere
        return samples
