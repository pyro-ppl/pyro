from __future__ import absolute_import, division, print_function

from .messenger import Messenger
from .trace_struct import Trace


class ConditionMessenger(Messenger):
    """
    Adds values at observe sites to condition on data and override sampling
    """
    def __init__(self, data):
        """
        :param data: a dict or a Trace

        Constructor. Doesn't do much, just stores the stochastic function
        and the data to condition on.
        """
        super(ConditionMessenger, self).__init__()
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
            assert not msg["is_observed"], \
                "should not change values of existing observes"
            if isinstance(self.data, Trace):
                msg["value"] = self.data.nodes[name]["value"]
            else:
                msg["value"] = self.data[name]
            msg["is_observed"] = True
        return None

    def _pyro_param(self, msg):
        return None
