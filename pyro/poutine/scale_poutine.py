import pyro
import torch
from torch.autograd import Variable

from .poutine import Poutine


class ScalePoutine(Poutine):
    """
    score-rescaling Poutine
    Subsampling means we have to rescale pdfs inside map_data
    This poutine handles the rescaling because it wouldn't fit in Poutine
    """
    def __init__(self, fn, scale):
        """
        Constructor
        """
        self.scale = scale
        super(ScalePoutine, self).__init__(fn)

    def _block_down(self, msg):
        """
        Dont continue down the stack, we're good here
        """
        return True

    def down(self, msg):
        """
        ScalePoutine has a side effect - pass the scale down the stack via msg
        """
        msg["scale"] = self.scale
        return super(ScalePoutine, self).down(msg)

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        """
        Rescale the scorer of the stochastic function passed to sample
        """
        msg["scale"] = self.scale
        return super(ScalePoutine, self)._pyro_sample(
            msg, name, fn, *args, **kwargs)

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        """
        Rescale the scorer of the stochastic function passed to observe
        """
        msg["scale"] = self.scale
        return super(ScalePoutine, self)._pyro_observe(
            msg, name, fn, obs, *args, **kwargs)
