import pyro
import torch

from pyro.infer import Trace
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
        self.sites = sites
        self.guide_trace = guide_trace


    def _pyro_sample(self, name, fn, *args, **kwargs):
        """
        Return the sample in the guide trace
        
        Expected behavior list:
        Case 1: self.sites is None and self.guide_trace is empty
        --> sample from model and store in trace
        Case 2: self.sites is None and self.guide_trace is non-empty
        --> replay sample from guide and store in trace
        Case 3: name in self.sites and self.guide_trace is empty
        --> sample from model and store in trace
        Case 4: name in self.sites and name not in self.guide_trace
        --> ambiguous - raise error, or sample from model and store in trace?
        Case 5: name in self.sites and name in self.guide_trace
        --> replay sample from guide and store in trace
        Case 6: name not in self.sites and self.guide_trace is empty
        --> sample from model but dont store
        Case 7: name not in self.sites and name in self.guide_trace
        --> ambiguous - raise error, or replay sample from guide but dont store?
        
        Any behavior cases missing?

        """
        assert(name in self.guide_trace)
        assert(self.guide_trace[name]["type"] == "sample")
        return self.guide_trace[name]


    def _pyro_map_data(self, data, fn):
        """
        Use the batch indices from the guide trace
        """
        raise NotImplementedError("havent finished this yet")
