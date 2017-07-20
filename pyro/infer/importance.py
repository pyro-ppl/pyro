import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine
from pyro.infer import TracePosterior


# XXX what should be the base class here?
class Importance(TracePosterior):
    """
    A new implementation of importance sampling
    """
    def __init__(self, model, guide=None, samples=None):
        """
        Constructor
        TODO proper docs etc
        """
        super(Importance, self).__init__()
        if samples is None:
            samples = 10
        if guide is None:
            # propose from the prior
            guide = poutine.block(model, hide_types=["observe"])
        self.samples = samples
        self.model = model
        self.guide = guide

    def _traces(self, *args, **kwargs):
        """
        make trace posterior histogram (unnormalized)
        """
        traces = []
        for i in range(self.samples):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)
            log_weight = model_trace.log_pdf() - guide_trace.log_pdf()
            yield (model_trace, log_weight)
