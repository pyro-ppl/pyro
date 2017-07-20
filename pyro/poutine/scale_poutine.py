import pyro
import torch
import copy
from torch.autograd import Variable

from .poutine import Poutine


class ScalePoutine(Poutine):
    """
    Rescaling poutine
    p = poutine.scale(f, scale=N/M)(*args, **kwargs)

    Expected behavior:
    all sample and observe log_pdf functions are monkeypatched to rescale with scale
    """
    def __init__(self, fn, scale=None):
        """
        Constructor
        """
        super(ScalePoutine, self).__init__(fn)
        self.transparent = False
        if scale is None:
            scale = 1.0
        if not isinstance(scale, Variable):
            scale = Variable(torch.Tensor([float(scale)]), requires_grad=False)
        self.scale = scale

    def rescale(self, fn):
        """
        Rescale a stochastic function's log_pdf method with self.scale
        """
        if hasattr(fn, "log_pdf"):
            old_log_pdf = fn.log_pdf

            def new_log_pdf(*args, **kwargs):
                return self.scale * old_log_pdf(*args, **kwargs)

            new_fn = copy.copy(fn)  # XXX incorrect?
            new_fn.log_pdf = new_log_pdf
            return new_fn
        else:
            # XXX should raise an error here?
            return fn

    def _pyro_sample(self, prev_val, name, _fn, *args, **kwargs):
        """
        Sample
        """
        fn = self.rescale(_fn)
        return super(ScalePoutine, self)._pyro_sample(prev_val, name, fn,
                                                      *args, **kwargs)

    def _pyro_observe(self, prev_val, name, _fn, val, *args, **kwargs):
        """
        observe
        """
        fn = self.rescale(_fn)
        return super(ScalePoutine, self)._pyro_observe(prev_val, name, fn, val,
                                                       *args, **kwargs)
