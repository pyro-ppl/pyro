import six
import torch
from torch.autograd import Variable
from collections import OrderedDict

import pyro
from pyro.infer.abstract_infer import AbstractInfer
import pyro.poutine as poutine


# XXX what should be the base class here?
class Importance(AbstractInfer):
    """
    A new implementation of importance sampling
    """
    def __init__(self, model, guide):
        """
        Constructor
        TODO proper docs etc
        """
        super(Importance, self).__init__()
        self.model = TracePoutine(model)
        self.guide = TracePoutine(guide)

    def runner(self, num_samples, *args, **kwargs):
        """
        main control loop
        TODO proper docs
        """
        # for each requested sample, we must:
        for i in range(num_samples):
            # sample from the guide
            # replay the model from the guide
            # sample from the model and store the return value
            # compute the log-joints from the model and guide traces
            # compute the unnormalized log-weight as logp - logq
            pass
        # normalize the weights (logsumexp?)
        # return list(?) of samples and normalized weights to be consumed elsewhere
        raise NotImplementedError("importance sampling not done yet!!")
    
