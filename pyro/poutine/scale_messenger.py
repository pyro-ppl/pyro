# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Union

import torch

from pyro.poutine.messenger import Messenger
from pyro.poutine.util import is_validation_enabled

if TYPE_CHECKING:
    from pyro.poutine.runtime import Message


class ScaleMessenger(Messenger):
    """
    Given a stochastic function with some sample statements and a positive
    scale factor, scale the score of all sample and observe sites in the
    function.

    Consider the following Pyro program:

        >>> def model(x):
        ...     s = pyro.param("s", torch.tensor(0.5))
        ...     pyro.sample("z", dist.Normal(x, s), obs=torch.tensor(1.0))

    ``scale`` multiplicatively scales the log-probabilities of sample sites:

        >>> scaled_model = pyro.poutine.scale(model, scale=0.5)
        >>> scaled_tr = pyro.poutine.trace(scaled_model).get_trace(0.0)
        >>> unscaled_tr = pyro.poutine.trace(model).get_trace(0.0)
        >>> bool((scaled_tr.log_prob_sum() == 0.5 * unscaled_tr.log_prob_sum()).all())
        True

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param scale: a positive scaling factor
    :returns: stochastic function decorated with a :class:`~pyro.poutine.scale_messenger.ScaleMessenger`
    """

    def __init__(self, scale: Union[float, torch.Tensor]) -> None:
        if isinstance(scale, torch.Tensor):
            if is_validation_enabled() and not (scale > 0).all():
                raise ValueError(
                    "Expected scale > 0 but got {}. ".format(scale)
                    + "Consider using poutine.mask() instead of poutine.scale()."
                )
        elif not (scale > 0):
            raise ValueError("Expected scale > 0 but got {}".format(scale))
        super().__init__()
        self.scale = scale

    def _process_message(self, msg: "Message") -> None:
        msg["scale"] = self.scale * msg["scale"]
