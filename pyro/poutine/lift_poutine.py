import pyro
from .poutine import Poutine


class LiftPoutine(Poutine):
    """
    Implements the param->sample lifting operation that turns params into rvs
    """
    def __init__(self, fn, prior):
        """
        constructor
        """
        self.prior = prior
        self.transparent = True
        super(LiftPoutine, self).__init__(fn)

    def _pyro_param(self, prev_val, name, *args, **kwargs):
        """
        prototype override of param->sample
        """
        prev_val["type"] = "sample"
        prev_val["fn"] = self.prior
        return self._pyro_sample(prev_val, name, self.prior, *args, **kwargs)
