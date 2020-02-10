# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import numbers
import warnings

import torch

from .messenger import Messenger
from .runtime import apply_stack


class DoMessenger(Messenger):
    """
    Given a stochastic function with some sample statements
    and a dictionary of values at names,
    set the return values of those sites equal to the values
    as if they were hard-coded to those values
    and introduce fresh sample sites with the same names
    whose values do not propagate.

    Composes freely with :func:`~pyro.poutine.handlers.condition`
    to represent counterfactual distributions over potential outcomes.
    See Single World Intervention Graphs [1] for additional details and theory.

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     z = pyro.sample("z", dist.Normal(x, s))
        ...     return z ** 2

    To intervene with a value for site `z`, we can write

        >>> intervened_model = pyro.poutine.do(model, data={"z": torch.tensor(1.)})

    This is equivalent to replacing `z = pyro.sample("z", ...)` with
    `z = torch.tensor(1.)`
    and introducing a fresh sample site pyro.sample("z", ...) whose value is not used elsewhere.

    References

    [1] `Single World Intervention Graphs: A Primer`,
        Thomas Richardson, James Robins

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param data: a ``dict`` mapping sample site names to interventions
    :returns: stochastic function decorated with a :class:`~pyro.poutine.do_messenger.DoMessenger`
    """
    def __init__(self, data):
        super().__init__()
        self.data = data
        self._intervener_id = str(id(self))

    def _pyro_sample(self, msg):
        if msg.get('_intervener_id', None) != self._intervener_id and \
                self.data.get(msg['name']) is not None:

            if msg.get('_intervener_id', None) is not None:
                warnings.warn(
                    "Attempting to intervene on variable {} multiple times,"
                    "this is almost certainly incorrect behavior".format(msg['name']),
                    RuntimeWarning)

            msg['_intervener_id'] = self._intervener_id

            # split node, avoid reapplying self recursively to new node
            new_msg = msg.copy()
            apply_stack(new_msg)

            # apply intervention
            intervention = self.data[msg['name']]
            msg['name'] = msg['name'] + "__CF"  # mangle old name

            if isinstance(intervention, (numbers.Number, torch.Tensor)):
                msg['value'] = intervention
                msg['is_observed'] = True
                msg['stop'] = True
            else:
                raise NotImplementedError(
                    "Interventions of type {} not implemented (yet)".format(type(intervention)))

        return None
