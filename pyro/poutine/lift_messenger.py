from __future__ import absolute_import, division, print_function

import warnings

from pyro import params
from pyro.distributions import Distribution
from pyro.poutine.util import is_validation_enabled

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
        self._samples_cache = {}

    def __enter__(self):
        self._samples_cache = {}
        if is_validation_enabled() and isinstance(self.prior, dict):
            self._param_hits = set()
            self._param_misses = set()
        return super(LiftMessenger, self).__enter__()

    def __exit__(self, *args, **kwargs):
        self._samples_cache = {}
        if is_validation_enabled() and isinstance(self.prior, dict):
            extra = set(self.prior) - self._param_hits
            if extra:
                warnings.warn(
                    "pyro.module prior did not find params ['{}']. "
                    "Did you instead mean one of ['{}']?"
                    .format("', '".join(extra), "', '".join(self._param_misses)))
        return super(LiftMessenger, self).__exit__(*args, **kwargs)

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
                msg["args"] = msg["args"][1:]
                if isinstance(msg['fn'], Distribution):
                    msg["args"] = ()
                    msg["kwargs"] = {}
                    msg["infer"] = {}
                if is_validation_enabled():
                    self._param_hits.add(param_name)
            else:
                if is_validation_enabled():
                    self._param_misses.add(param_name)
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
            msg["args"] = msg["args"][1:]
        else:
            # otherwise leave as is
            return None
        msg["type"] = "sample"
        if name in self._samples_cache:
            # Multiple pyro.param statements with the same
            # name. Block the site and fix the value.
            msg['value'] = self._samples_cache[name]['value']
            msg["is_observed"] = True
            msg["stop"] = True
        else:
            self._samples_cache[name] = msg
            msg["is_observed"] = False
        return self._pyro_sample(msg)
