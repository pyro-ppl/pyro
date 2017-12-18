from __future__ import absolute_import, division, print_function

from .poutine import Poutine
from .trace import Trace


class ConditionPoutine(Poutine):
    """
    Adds values at observe sites to condition on data and override sampling
    """
    def __init__(self, fn, data):
        """
        :param fn: a stochastic function (callable containing pyro primitive calls)
        :param data: a dict or a Trace

        Constructor. Doesn't do much, just stores the stochastic function
        and the data to condition on.
        """
        self.data = data
        super(ConditionPoutine, self).__init__(fn)

    def _prepare_site(self, msg):
        """
        :param msg: current message at a trace site
        :returns: the updated message at the same trace site

        If we have data at this site, don't sample from the site function
        """
        if msg["name"] in self.data and msg["type"] == "sample":
            msg["done"] = True
        return msg

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
        if msg["is_observed"]:
            assert name not in self.data, \
                "should not change values of existing observes"

        if name in self.data:
            msg["done"] = False
            if isinstance(self.data, Trace):
                msg["value"] = self.data.nodes[name]["value"]
            else:
                msg["value"] = self.data[name]
            msg["is_observed"] = True
            return super(ConditionPoutine, self)._pyro_sample(msg)
        else:
            return super(ConditionPoutine, self)._pyro_sample(msg)
