import pyro

from .poutine import Poutine
from .scale_poutine import ScalePoutine
from .lambda_poutine import LambdaPoutine


class ReplayPoutine(Poutine):
    """
    Poutine for replaying from an existing execution trace
    """

    def __init__(self, fn, guide_trace, sites=None):
        """
        Constructor.
        """
        super(ReplayPoutine, self).__init__(fn)
        assert guide_trace is not None, "must provide guide_trace"
        self.guide_trace = guide_trace
        # case 1: no sites
        if sites is None:
            self.sites = {site: site for site in guide_trace.keys()}
        # case 2: sites is a list/tuple/set
        elif isinstance(sites, (list, tuple, set)):
            self.sites = {site: site for site in sites}
        # case 3: sites is a dict
        elif isinstance(sites, dict):
            self.sites = sites
        # otherwise, something is wrong
        # XXX one other possible case: sites is a trace?
        else:
            raise TypeError(
                "unrecognized type {} for sites".format(str(type(sites))))

    def down(self, msg):
        """
        Pass indices down at a map_data
        """
        if msg["type"] == "map_data":
            if msg["name"] in self.guide_trace:
                assert self.guide_trace[msg["name"]]["type"] == "map_data", \
                    msg["name"] + " is not a map_data in the guide_trace"
                msg["indices"] = self.guide_trace[msg["name"]]["indices"]
                msg["batch_size"] = self.guide_trace[msg["name"]]["batch_size"]
                msg["batch_dim"] = self.guide_trace[msg["name"]]["batch_dim"]

        barrier = self._block_down(msg)
        return msg, barrier

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        """
        Return the sample in the guide trace when appropriate
        """
        # case 1: dict, positive: sample from guide
        if name in self.sites:
            g_name = self.sites[name]
            assert g_name in self.guide_trace, \
                "{} in sites but {} not in trace".format(name, g_name)
            assert self.guide_trace[g_name]["type"] == "sample", \
                "site {} must be sample in guide_trace".format(g_name)
            return self.guide_trace[g_name]["value"]
        # case 2: dict, negative: sample from model
        elif name not in self.sites:
            return super(ReplayPoutine, self)._pyro_sample(msg, name, fn, *args, **kwargs)
        else:
            raise ValueError(
                "something went wrong with replay conditions at site " + name)

    def _pyro_map_data(self, msg, name, data, fn, batch_size=None, batch_dim=0):
        """
        Scale
        """
        print "replay poutine map data: %s" % name
        scale = pyro.util.get_batch_scale(data, batch_size, batch_dim)
        msg['lambda_instqlled'] = True
        return super(ReplayPoutine, self)._pyro_map_data(msg, name, data,
							 LambdaPoutine(ScalePoutine(fn, scale), name),
                                                         batch_size=batch_size,
                                                         batch_dim=batch_dim)

