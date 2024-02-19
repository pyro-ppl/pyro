# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from pyro.poutine.messenger import Messenger

if TYPE_CHECKING:
    from pyro.poutine.runtime import Message


class UnconditionMessenger(Messenger):
    """
    Messenger to force the value of observed nodes to be sampled from their
    distribution, ignoring observations.
    """

    def __init__(self) -> None:
        super().__init__()

    def _pyro_sample(self, msg: "Message") -> None:
        """
        :param msg: current message at a trace site.

        Samples value from distribution, irrespective of whether or not the
        node has an observed value.
        """
        if msg["is_observed"]:
            msg["is_observed"] = False
            assert msg["infer"] is not None
            msg["infer"]["was_observed"] = True
            msg["infer"]["obs"] = msg["value"]
            msg["value"] = None
            msg["done"] = False
