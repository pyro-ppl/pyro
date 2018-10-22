from __future__ import absolute_import, division, print_function

from .broadcast_messenger import BroadcastMessenger
from .subsample_messenger import SubsampleMessenger


class PlateMessenger(SubsampleMessenger):
    """
    Swiss army knife of broadcasting amazingness:
    combines shape inference, independence annotation, and subsampling
    """
    def _process_message(self, msg):
        super(PlateMessenger, self)._process_message(msg)
        return BroadcastMessenger._pyro_sample(msg)

    def __enter__(self):
        super(PlateMessenger, self).__enter__()
        if self._vectorized and self._indices is not None:
            return self.indices
        return None
