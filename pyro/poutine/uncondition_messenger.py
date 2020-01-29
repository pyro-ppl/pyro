# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from .messenger import Messenger


class UnconditionMessenger(Messenger):
    """
    Messenger to force the value of observed nodes to be sampled from their
    distribution, ignoring observations.
    """
    def __init__(self):
        super().__init__()

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.

        Samples value from distribution, irrespective of whether or not the
        node has an observed value.
        """
        if msg["is_observed"]:
            msg["is_observed"] = False
            msg["infer"]["was_observed"] = True
            msg["infer"]["obs"] = msg["value"]
            msg["value"] = None
            msg["done"] = False
        return None
