import pyro
import torch

from pyro.poutine import Trace
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

    def _pyro_sample(self, prev_val, name, fn, *args, **kwargs):
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
            return fn(*args, **kwargs)
        else:
            raise ValueError(
                "something went wrong with replay conditions at site " + name)

    def _pyro_map_data(self, prev_val, name, data, fn, batch_size=None):
        """
        Use the batch indices from the guide trace
        """
        if batch_size is None:
            batch_size = 0

        assert batch_size >= 0, "cannot have negative batch sizes"
        if name in self.guide_trace:
            assert self.guide_trace[name]["type"] == "map_data", \
                name + " is not a map_data in the guide_trace"
            batch_size = self.guide_trace[name]["batch_size"]
            fn.__map_data_scale = self.guide_trace[name]["scale"]
            fn.__map_data_indices = self.guide_trace[name]["indices"]

        ret = super(ReplayPoutine, self)._pyro_map_data(prev_val, name, data, fn,
                                                        batch_size=batch_size)
        return ret
