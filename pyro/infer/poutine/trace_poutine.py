import pyro
import torch
from torch.autograd import Variable
#from torch.multiprocessing import Pool

from .poutine import Poutine


class TracePoutine(Poutine):
    """
    Execution trace poutine.

    A TracePoutine records the input and output to every pyro primitive
    and stores them as a Site() in a Trace().
    This should, in theory, be sufficient information for every inference algorithm
    (along with the implicit computational graph in the Variables?)

    We can also use this for visualization.
    """
    def __init__(self, fn, *args, **kwargs):
        """
        Constructor
        """
        super(TracePoutine, self).__init__(fn, *args, **kwargs)
        self.trace = Trace()

    def __call__(self, *args, **kwargs):
        """
        aa
        """
        ret = super(TracePoutine, self).__call__(*args, **kwargs)
        self.trace.add_return(ret, *args, **kwargs)
        return self.trace

    def _pyro_map_data(self, name, *args, **kwargs):
        """
        Trace map_data
        """
        raise NotImplementedError("still working out proper semantics")
        
    def _pyro_sample(self, name, dist, *args, **kwargs):
        """
        sample
        TODO docs
        """
        assert(name not in self.trace)
        val = super(TracePoutine, self)._pyro_sample(name, dist, *args, **kwargs)
        # XXX not correct arguments
        self.trace.add_sample(name, val, dist, *args, **kwargs)
        return val

    def _pyro_observe(self, name, dist, val, *args, **kwargs):
        """
        observe
        TODO docs
        
        Expected behavior:
        TODO
        """
        # make sure the site name is unique
        assert(name not in self.trace)
        val = super(TracePoutine, self)._pyro_observe(name, dist, val,
                                                      *args, **kwargs)
        # XXX not correct arguments?
        self.trace.add_observe(name, dist, val, *args, **kwargs)


    def _pyro_param(self, name, *args, **kwargs):
        """
        param
        TODO docs
        
        Expected behavior:
        TODO
        """
        retrieved = super(TracePoutine, self)._pyro_param(name, *args, **kwargs)
        self.trace.add_param(name, retrieved, *args, **kwargs)
        return retrieved


       
