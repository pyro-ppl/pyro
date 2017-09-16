from pyro import params
from .poutine import Poutine


class LiftPoutine(Poutine):
    """
    Implements the param->sample lifting operation that turns params into rvs
    """
    # XXX docs
    def __init__(self, fn, prior):
        """
        constructor
        """
        self.prior = prior
        super(LiftPoutine, self).__init__(fn)

    def _block_down(self, msg):
        if msg["type"] == "param":
            return True
        return False

    def _pyro_param(self, msg, name, *args, **kwargs):
        """
        prototype override of param->sample
        """
        msg["type"] = "sample"
        param_name = params.user_param_name(name)
        if isinstance(self.prior, dict):
            if param_name in self.prior.keys():
                msg["fn"] = self.prior[param_name]
        else:
            # prior is a distribution
            msg["fn"] = self.prior
        msg["scale"] = 1.0
        return self._pyro_sample(msg, name, msg["fn"], *args, **kwargs)
