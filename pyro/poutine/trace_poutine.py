import pyro
import torch
import pdb
from .poutine import Poutine
from pyro.infer.trace import Trace


class TracePoutine(Poutine):
    """
    Execution trace poutine.

    A TracePoutine records the input and output to every pyro primitive
    and stores them as a site in a Trace().
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
        # Have to override this to change return type
        ret = super(TracePoutine, self).__call__(*args, **kwargs)
        return self.trace
        
    def _enter_poutine(self, *args, **kwargs):
        """
        Register the input arguments in the trace upon entry
        """
        super(TracePoutine, self)._enter_poutine(*args, **kwargs)
        self.trace = Trace()
        self.trace.add_args((args, kwargs))

    def _exit_poutine(self, ret_val, *args, **kwargs):
        """
        Register the return value from the function on exit
        """
        self.trace.add_return(ret_val, *args, **kwargs)
    
    def _pyro_sample(self, prev_val, name, dist, *args, **kwargs):
        """
        sample
        TODO docs
        """
        assert(name not in self.trace)
        val = super(TracePoutine, self)._pyro_sample(prev_val, name, dist,
                                                     *args, **kwargs)
        # XXX not correct arguments
        self.trace.add_sample(name, val, dist, *args, **kwargs)
        return val

    def _pyro_observe(self, prev_val, name, fn, obs, *args, **kwargs):
        """
        observe
        TODO docs
        
        Expected behavior:
        TODO
        """
        # make sure the site name is unique
        assert(name not in self.trace)
        val = super(TracePoutine, self)._pyro_observe(prev_val, name, fn, obs,
                                                      *args, **kwargs)
        self.trace.add_observe(name, val, fn, obs, *args, **kwargs)
        return val

    def _pyro_param(self, prev_val, name, *args, **kwargs):
        """
        param
        TODO docs
        
        Expected behavior:
        TODO
        """
        retrieved = super(TracePoutine, self)._pyro_param(prev_val, name,
                                                          *args, **kwargs)
        self.trace.add_param(name, retrieved, *args, **kwargs)
        return retrieved

    # def _pyro_map_data(self, prev_val, name, *args, **kwargs):
    #     """
    #     Trace map_data
    #     """
    #     raise NotImplementedError("still working out proper semantics")
      
