import pyro

from .poutine import Poutine
from .scale_poutine import ScalePoutine


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
        if msg["name"] in self.sites:
            if msg["type"] == "map_data":
                guide_type = self.guide_trace[msg["name"]]["type"]
                assert self.guide_trace[msg["name"]]["type"] == "map_data", \
                    "{} is {}, not map_data, in guide_trace".format(msg["name"],
                                                                    guide_type)
                msg["indices"] = self.guide_trace[msg["name"]]["indices"]
                msg["batch_size"] = self.guide_trace[msg["name"]]["batch_size"]
                msg["batch_dim"] = self.guide_trace[msg["name"]]["batch_dim"]

            # dont reexecute
            if msg["type"] == "sample":
                msg["done"] = True

        return msg

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
            msg["done"] = True
            return self.guide_trace[g_name]["value"]
        # case 2: dict, negative: sample from model
        elif name not in self.sites:
            return super(ReplayPoutine, self)._pyro_sample(msg, name, fn, *args, **kwargs)
        else:
            raise ValueError(
                "something went wrong with replay conditions at site " + name)

    def _pyro_map_data(self, msg, name, data, fn, batch_size=None, batch_dim=0):
        """
        Use the batch indices from the guide trace, already provided by down
        So all we need to do here is apply a ScalePoutine as in TracePoutine
        """
        scale = pyro.util.get_batch_scale(data, batch_size, batch_dim)
        return super(ReplayPoutine, self)._pyro_map_data(msg, name, data,
                                                         ScalePoutine(fn, scale),
                                                         batch_size=batch_size,
                                                         batch_dim=batch_dim)
