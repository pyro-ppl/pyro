import pyro
import torch

from .poutine import Poutine
from pyro.infer.trace import Trace


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
        Main logic; where the function is actually called
        """
        # Have to override this to log inputs and outputs and change return type
        self.trace.add_args((args, kwargs))
        ret = super(TracePoutine, self).__call__(*args, **kwargs)
        self.trace.add_return(ret, *args, **kwargs)
        return self.trace
        

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


    def _pyro_observe(self, name, fn, obs, *args, **kwargs):
        """
        observe
        TODO docs
        
        Expected behavior:
        TODO
        """
        # make sure the site name is unique
        assert(name not in self.trace)
        val = super(TracePoutine, self)._pyro_observe(name, fn, obs, *args, **kwargs)
        # XXX not correct arguments?
        self.trace.add_observe(name, val, fn, obs, *args, **kwargs)


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


     def _pyro_map_data(self, name, *args, **kwargs):
        """
        Trace map_data
        """
        raise NotImplementedError("still working out proper semantics")
      
