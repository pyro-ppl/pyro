import pyro
import torch

from pyro.infer.trace import Trace
from .poutine import Poutine


class ReplayPoutine(Poutine):
    """
    Poutine for replaying from an existing execution trace
    """
    def __init__(self, fn, guide_trace, sites=None):
        """
        Constructor.
        """
        super(ReplayPoutine, self).__init__(fn)
        self.transparent = False
        self.guide_trace = guide_trace
        self.all_sites = False
        # case 1: no sites
        if sites is None:
            self.all_sites = True
        # case 3: sites is a list/tuple/set
        elif isinstance(sites, (list, tuple, set)):
            self.sites = {site: site for site in sites}
        # case 4: sites is a dict
        elif isinstance(sites, dict):
            self.sites = sites
        # otherwise, something is wrong
        # XXX one other possible case: sites is a trace?
        else:
            raise TypeError(
                "unrecognized type {} for sites".format(str(type(sites))))

    def _pyro_sample(self, prev_val, name, fn, *args, **kwargs):
        """
        Return the sample in the guide trace when appropriate
        """
        # case 1: all_sites
        if self.all_sites:
            # some sanity checks
            assert(name in self.guide_trace)
            assert(self.guide_trace[name]["type"] == "sample")
            return self.guide_trace[name]["value"]
        # case 3: dict
        if self.sites is not None:
            # case 3a: dict, positive: sample from guide
            if name in self.sites:
                g_name = self.sites[name]
                assert(g_name in self.guide_trace)
                assert(self.guide_trace[g_name]["type"] == "sample")
                return self.guide_trace[g_name]["value"]
            # case 3b: dict, negative: sample from model
            elif name not in self.sites:
                return fn(*args, **kwargs)
            else:
                raise ValueError(
                    "something went wrong with replay conditions at site " + name)

    # def _pyro_map_data(self, prev_val, name, data, fn):
    #     """
    #     Use the batch indices from the guide trace
    #     """
    #     raise NotImplementedError("havent finished this yet")
