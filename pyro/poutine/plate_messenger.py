# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from .broadcast_messenger import BroadcastMessenger
from .subsample_messenger import SubsampleMessenger


class PlateMessenger(SubsampleMessenger):
    """
    Swiss army knife of broadcasting amazingness:
    combines shape inference, independence annotation, and subsampling
    """
    def _process_message(self, msg):
        super()._process_message(msg)
        return BroadcastMessenger._pyro_sample(msg)

    def __enter__(self):
        super().__enter__()
        if self._vectorized and self._indices is not None:
            return self.indices
        return None
