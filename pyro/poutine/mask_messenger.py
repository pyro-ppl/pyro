# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from .messenger import Messenger


class MaskMessenger(Messenger):
    """
    Given a stochastic function with some batched sample statements and
    masking tensor, mask out some of the sample statements elementwise.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param torch.BoolTensor mask: a ``{0,1}``-valued masking tensor
        (1 includes a site, 0 excludes a site)
    :returns: stochastic function decorated with a :class:`~pyro.poutine.scale_messenger.MaskMessenger`
    """

    def __init__(self, mask):
        if isinstance(mask, torch.Tensor):
            if mask.dtype != torch.bool:
                raise ValueError(
                    "Expected mask to be a BoolTensor but got {}".format(type(mask))
                )

        elif mask not in (True, False):
            raise ValueError(
                "Expected mask to be a boolean but got {}".format(type(mask))
            )
        super().__init__()
        self.mask = mask

    def _process_message(self, msg):
        msg["mask"] = self.mask if msg["mask"] is None else msg["mask"] & self.mask
        return None
