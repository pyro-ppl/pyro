import pyro
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

    def _prepare_site(self, msg):
        name = msg["name"]
        param_name = params.user_param_name(name)
        if isinstance(self.prior, dict) and param_name in self.prior.keys():
            if msg["type"] == "param":
                msg["done"] = True
        return msg

    def _pyro_param(self, msg):
        """
        prototype override of param->sample
        """
        name = msg["name"]
        param_name = params.user_param_name(name)
        if isinstance(self.prior, dict):
            if param_name in self.prior.keys():
                msg["fn"] = self.prior[param_name]
            else:
                return super(LiftPoutine, self)._pyro_param(msg)
        elif callable(self.prior):
            # prior is a distribution
            msg["fn"] = self.prior
        else:
            # otherwise leave as is
            return super(LiftPoutine, self)._pyro_param(msg)
        msg["type"] = "sample"
        msg["done"] = False
        return self._pyro_sample(msg)
