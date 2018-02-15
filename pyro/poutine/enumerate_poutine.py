from __future__ import absolute_import, division, print_function

from .poutine import Messenger, Poutine
from .trace import Trace


class EnumerateMessenger(Messenger):
    """
    Adds values at observe sites to condition on data and override sampling
    """
    def __init__(self, first_available_dim=0):
        """
        :param int dim: The target dimension indexing the enumerated samples.

        Constructor. Doesn't do much, just stores the stochastic function
        and the data to condition on.
        """
        super(EnumerateMessenger, self).__init__()
        self.next_available_dim = next_available_dim

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.
        """
        if msg["done"]:
            return
        if msg["infer"].get("enumerate", None) == "parallel":
            dim = self.next_available_dim
            self.next_available_dim += 1

            value = msg["fn"].enumerate_support()
            value = value.expand(TODO(dim))
            msg["value"] = value
            msg["done"] = True


class EnumeratePoutine(Poutine):
    """
    Adds values at observe sites to condition on data and override sampling
    """
    def __init__(self, fn, first_available_dim=0):
        """
        :param fn: a stochastic function (callable containing pyro primitive calls)
        :param data: a dict or a Trace

        Constructor. Doesn't do much, just stores the stochastic function
        and the data to condition on.
        """
        super(EnumeratePoutine, self).__init__(EnumerateMessenger(data), fn)
