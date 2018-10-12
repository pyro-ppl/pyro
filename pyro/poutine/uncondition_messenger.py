from __future__ import absolute_import, division, print_function

from .messenger import Messenger


class UnconditionMessenger(Messenger):
    """
    Messenger to force the value of observed nodes to be sampled from their
    distribution, ignoring observations.
    """
    def __init__(self):
        super(UnconditionMessenger, self).__init__()

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.

        Samples value from distribution, irrespective of whether or not the
        node has an observed value.
        """
        msg["value"] = msg["fn"](*msg["args"], **msg["kwargs"])
        return None
