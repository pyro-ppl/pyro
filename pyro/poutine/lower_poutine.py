from __future__ import absolute_import, division, print_function

from torch.autograd import Variable
import pyro  # XXX remove this cyclic dependency
from pyro.params import _PYRO_PARAM_STORE

from .poutine import Poutine


class LowerPoutine(Poutine):
    """
    Poutine which "lowers" sample sites to param + observe sites.

    Given a stochastic function with sample calls from a prior, creates a
    stochastic function where each sample site is replaced by a param site
    (denoting the parameter) plus an observe site (denoting the prior).

    :param fn: stochastic function
    """
    def _pyro_sample(self, msg):
        if msg["is_observed"]:
            return super(LowerPoutine, self)._pyro_sample(msg)
        param_name = msg["name"]
        msg["name"] = "prior.{}".format(param_name)
        if param_name in _PYRO_PARAM_STORE:
            # Treat existing param value as an observation.
            msg["is_observed"] = True
            msg["value"] = _PYRO_PARAM_STORE.get_param(param_name)
            value = super(LowerPoutine, self)._pyro_sample(msg)
        else:
            # Sample an initial param value from the prior.
            value = super(LowerPoutine, self)._pyro_sample(msg)
            value = Variable(value.data, requires_grad=True)
            msg["value"] = value
            msg["is_observed"] = True
        return pyro.param(param_name, value)
