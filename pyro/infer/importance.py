import torch
from torch.autograd import Variable

import pyro
import pyro.poutine as poutine
from pyro.distributions import Categorical


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

    def _dist(self, *args, **kwargs):
        """
        make trace posterior distribution
        """
        traces = []
        log_weights = []
        for i in range(self.samples):
            guide_trace = poutine.trace(self.guide)(*args, **kwargs)
            model_trace = poutine.trace(
                poutine.replay(self.model, guide_trace))(*args, **kwargs)
            traces.append(model_trace)
            log_weights.append(model_trace.log_pdf() - guide_trace.log_pdf())

        log_ps = Variable(torch.Tensor(log_weights))
        log_ps = log_ps - pyro.util.log_sum_exp(log_weights)
        return Categorical(ps=torch.exp(log_ps), vs=traces)       

    def sample(self, *args, **kwargs):
        """
        sample from trace posterior
        """
        return self._dist(*args, **kwargs).sample()

    def log_pdf(self, val, *args, **kwargs):
        return self._dist(*args, **kwargs)

    def log_z(self, *args, **kwargs):
        traces = self._dist(*args, **kwargs).vs
        log_z = 0.0
        for tr in traces:
            log_z = log_z + tr.log_pdf()
            guide_tr = poutine.trace(poutine.replay(self.guide, tr))(*args, **kwargs)
            log_z = log_z - guide_tr.log_pdf()
        return log_z / len(traces)
