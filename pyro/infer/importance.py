import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine


# XXX what should be the base class here?
class Importance(pyro.infer.abstract_infer.AbstractInfer):
    """
    A new implementation of importance sampling
    """
    def __init__(self, model, guide=None, samples=10):
        """
        Constructor
        TODO proper docs etc
        """
        super(Importance, self).__init__()
        self.samples = samples
        self.model = model
        if guide is None:
            # propose from the prior
            guide = poutine.block(model, hide_types=["observe"])
        self.guide = guide

    def _traces(self, *args, **kwargs):
        """
        make trace posterior histogram (unnormalized)
        """
        traces = []
        log_weights = []
        for i in range(self.samples):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)
            traces.append(model_trace)
            log_weights.append(model_trace.log_pdf() - guide_trace.log_pdf())
        return traces, log_weights
