from pyro import params
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
        Overrides the `pyro.param` call with samples sampled from the
        distribution specified in the prior. The prior can be a
        pyro.distributions object or a dict of distributions keyed
        on the param names. If the param name does not match the
        name the keys in the prior, that param name is unchanged.

        TODO: any stochastic fn
        """
        name = msg["name"]
        param_name = params.user_param_name(name)
        if isinstance(self.prior, dict):
            # prior is a dict of distributions
            if param_name in self.prior.keys():
                msg["fn"] = self.prior[param_name]
            else:
                return super(LiftPoutine, self)._pyro_param(msg)
        elif isinstance(self.prior, pyro.distributions.Distribution)
            # prior is a distribution
            msg["fn"] = self.prior
        elif callable(self.prior):
            # prior is a stochastic fn
            msg["stop"] = True
        else:
            # otherwise leave as is
            return super(LiftPoutine, self)._pyro_param(msg)
        msg["type"] = "sample"
        msg["done"] = False
        return self._pyro_sample(msg)
