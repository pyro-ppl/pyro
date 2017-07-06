import pyro
import torch

from pyro.infer import Trace
from .poutine import Poutine

class PivotPoutine(Poutine):
    """
    Poutine for replaying from an existing execution trace
    """
    def __init__(self, fn, guide_trace, pivot=None):
        """
        Constructor.
        """
        super(PivotPoutine, self).__init__(fn)
        self.guide_trace = guide_trace
        self.all_sites = False
        self.pivot_site = None
        # case 1: no sites or pivot
        if pivot is None:
            self.all_sites = True
        # case 2: pivot and no sites
        elif pivot is not None:
            self.pivot_site = pivot
        # otherwise, something is wrong
        # XXX one other possible case: sites is a trace?
        else:
            raise TypeError(
                "something went wrong with pivot site {}".format(str(pivot_site)))


    def _enter_poutine(self, *args, **kwargs):
        """
        Poutine entry
        """
        self.pivot_seen = False

    def _exit_poutine(self, *args, **kwargs):
        """
        Poutine exit
        """
        self.pivot_seen = False
        

    def _pyro_sample(self, name, fn, *args, **kwargs):
        """
        Return the sample in the guide trace when appropriate
        """
        # case 1: all_sites
        if self.all_sites:
            # some sanity checks
            assert(name in self.guide_trace)
            assert(self.guide_trace[name]["type"] == "sample")
            return self.guide_trace[name]["sample"]
        # case 2: pivot
        if self.pivot_site is not None:
            # case 2a: site is pivot
            if name == self.pivot_site:
                # XXX what to do here??
                pass
            # case 2b: pivot unseen: sample from guide
            elif not self.pivot_seen:
                assert(name in self.guide_trace)
                assert(self.guide_trace[name]["type"] == "sample")
                return self.guide_trace[name]["sample"] # XXX right entry?
            # case 2c: pivot seen: sample from model
            elif self.pivot_seen:
                return fn(*args, **kwargs)
            else:
                raise ValueError(
                    "something went wrong with replay conditions at site "+name)


    def _pyro_map_data(self, data, fn):
        """
        Use the batch indices from the guide trace
        """
        raise NotImplementedError("havent finished this yet")
