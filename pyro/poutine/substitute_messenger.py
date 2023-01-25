# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings

from pyro import params
from pyro.poutine.messenger import Messenger
from pyro.poutine.util import is_validation_enabled


class SubstituteMessenger(Messenger):
    """
    Given a stochastic function with param calls and a set of parameter values,
    create a stochastic function where all param calls are substituted with
    the fixed values.
    data should be a dict of names to values.
    Consider the following Pyro program:

        >>> def model(x):
        ...     a = pyro.param("a", torch.tensor(0.5))
        ...     x = pyro.sample("x", dist.Bernoulli(probs=a))
        ...     return x
        >>> substituted_model = pyro.poutine.substitute(model, data={"s": 0.3})

    In this example, site `a` will now have value `0.3`.
    :param data: dictionary of values keyed by site names.
    :returns: ``fn`` decorated with a :class:`~pyro.poutine.substitute_messenger.SubstituteMessenger`
    """

    def __init__(self, data):
        """
        :param data: values for the parameters.
        Constructor
        """
        super().__init__()
        self.data = data
        self._data_cache = {}

    def __enter__(self):
        self._data_cache = {}
        if is_validation_enabled() and isinstance(self.data, dict):
            self._param_hits = set()
            self._param_misses = set()
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        self._data_cache = {}
        if is_validation_enabled() and isinstance(self.data, dict):
            extra = set(self.data) - self._param_hits
            if extra:
                warnings.warn(
                    "pyro.module data did not find params ['{}']. "
                    "Did you instead mean one of ['{}']?".format(
                        "', '".join(extra), "', '".join(self._param_misses)
                    )
                )
        return super().__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        return None

    def _pyro_param(self, msg):
        """
        Overrides the `pyro.param` with substituted values.
        If the param name does not match the name the keys in `data`,
        that param value is unchanged.
        """
        name = msg["name"]
        param_name = params.user_param_name(name)

        if param_name in self.data.keys():
            msg["value"] = self.data[param_name]
            if is_validation_enabled():
                self._param_hits.add(param_name)
        else:
            if is_validation_enabled():
                self._param_misses.add(param_name)
            return None

        if name in self._data_cache:
            # Multiple pyro.param statements with the same
            # name. Block the site and fix the value.
            msg["value"] = self._data_cache[name]["value"]
        else:
            self._data_cache[name] = msg
