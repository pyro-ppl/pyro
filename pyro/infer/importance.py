import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine


# XXX what should be the base class here?
class Importance(pyro.infer.abstract_infer.AbstractInfer):
    """
    A new implementation of importance sampling
    """
    def __init__(self, model, guide):
        """
        Constructor
        TODO proper docs etc
        """
        super(Importance, self).__init__()
        self.model = model
        self.guide = guide

    def runner(self, num_samples, *args, **kwargs):
        """
        main control loop
        TODO proper docs
        """
        samples = []
        # for each requested sample, we must:
        for i in range(num_samples):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)
            samples.append([i, model_trace["_RETURN"]["value"],
                            model_trace.log_pdf() - guide_trace.log_pdf()])

        return samples
