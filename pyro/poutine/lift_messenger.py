from __future__ import absolute_import, division, print_function

from pyro import params
from pyro.distributions import Distribution

from .messenger import Messenger


class LiftMessenger(Messenger):
    """
    Messenger which "lifts" parameters to random samples.
    Given a stochastic function with param calls and a prior,
    creates a stochastic function where all param calls are
    replaced by sampling from prior.

    Prior should be a callable or a dict of names to callables.
    """

    def __init__(self, prior):
        """
        :param prior: prior used to lift parameters. Prior can be of type
                      dict, pyro.distributions, or a python stochastic fn

        Constructor
        """
        super(LiftMessenger, self).__init__()
        self.prior = prior

    def _pyro_sample(self, msg):
        return None

    def _pyro_param(self, msg):
        """
        Overrides the `pyro.param` call with samples sampled from the
        distribution specified in the prior. The prior can be a
        pyro.distributions object or a dict of distributions keyed
        on the param names. If the param name does not match the
        name the keys in the prior, that param name is unchanged.
        """
        name = msg["name"]
        param_name = params.user_param_name(name)
        if isinstance(self.prior, dict):
            # prior is a dict of distributions
            if param_name in self.prior.keys():
                msg["fn"] = self.prior[param_name]
                if isinstance(msg['fn'], Distribution):
                    msg["args"] = ()
                    msg["kwargs"] = {}
                    msg["infer"] = {}
            else:
                return None
        elif isinstance(self.prior, Distribution):
            # prior is a distribution
            msg["fn"] = self.prior
            msg["args"] = ()
            msg["kwargs"] = {}
            msg["infer"] = {}
        elif callable(self.prior):
            if not isinstance(self.prior, Distribution):
                # prior is a stochastic fn. block sample
                msg["stop"] = True
            msg["fn"] = self.prior
        else:
            # otherwise leave as is
            return None
        msg["type"] = "sample"
        msg["is_observed"] = False
        return self._pyro_sample(msg)
