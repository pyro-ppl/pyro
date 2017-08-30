import pyro
import torch

from .poutine import Poutine


class LiftPoutine(Poutine):
    """
    Implements the param->sample lifting operation that turns params into rvs
    Should block on down but not on up?
    """
    # XXX docs
    def __init__(self, fn, prior, *args, **kwargs):
        """
        constructor
        """
        self.prior = prior
        super(LiftPoutine, self).__init__(self, fn, *args, **kwargs)

    def _block_down(self, site_type, site_name):
        if site_type == "param":
            return True
        else:
            return False

    def _pyro_param(self, msg, name, *args, **kwargs):
        """
        prototype override of param->sample
        """
        msg["type"] = "sample"
        msg["fn"] = self.prior
        msg["scale"] = 1.0
        return self._pyro_sample(msg, name, self.prior, *args, **kwargs)
