# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from .messenger import Messenger
from .trace_struct import Trace


class ConditionMessenger(Messenger):
    """
    Given a stochastic function with some sample statements
    and a dictionary of observations at names,
    change the sample statements at those names into observes
    with those values.

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2

    To observe a value for site `z`, we can write

        >>> conditioned_model = pyro.poutine.condition(model, data={"z": torch.tensor(1.)})

    This is equivalent to adding `obs=value` as a keyword argument
    to `pyro.sample("z", ...)` in `model`.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param data: a dict or a :class:`~pyro.poutine.Trace`
    :returns: stochastic function decorated with a :class:`~pyro.poutine.condition_messenger.ConditionMessenger`
    """
    def __init__(self, data):
        """
        :param data: a dict or a Trace

        Constructor. Doesn't do much, just stores the stochastic function
        and the data to condition on.
        """
        super().__init__()
        self.data = data

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.

        If msg["name"] appears in self.data,
        convert the sample site into an observe site
        whose observed value is the value from self.data[msg["name"]].

        Otherwise, implements default sampling behavior
        with no additional effects.
        """
        name = msg["name"]

        if name in self.data:
            if isinstance(self.data, Trace):
                msg["value"] = self.data.nodes[name]["value"]
            else:
                msg["value"] = self.data[name]
            msg["is_observed"] = msg["value"] is not None
        return None
