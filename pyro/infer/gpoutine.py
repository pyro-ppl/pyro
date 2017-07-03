import six
import torch
from torch.autograd import Variable
import greenlet
from uuid import uuid4 as uuid

import pyro
from .trace import Site, Trace

# TODO import trace?

# XXX should be its own class instead of greenlet descendant?
class GPoutine(greenlet.greenlet):
    """
    Greenlet-based Poutine replacement
    TODO proper docs
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor
        TODO proper docs
        """
        super(GPoutine, self).__init__(*args, **kwargs)

    # XXX call needs to override switch, how should this be defined?
    def __call__(self, *args, **kwargs):
        """
        Runs the GPoutine to completion?
        """
        return self.switch(*args, **kwargs)
    
    def _enter_poutine(self, *args, **kwargs):
        """
        Called upon entry
        """
        pass

    def _exit_poutine(self, *args, **kwargs):
        """
        Called upon exit
        """
        pass

    def _get_current_stack(self):
        """
        Get current stack
        """
        pass

    def _push_stack(self):
        """
        Push primitives onto interpreter stack
        """
        pass

    def _pop_stack(self):
        """
        Pop primitives from interpreter stack
        """
        pass

    def _pyro_sample(self, *args, **kwargs):
        """
        Abstract entry point for sample
        """
        pass

    def _pyro_observe(self, *args, **kwargs):
        """
        Abstract entry point for observe
        """
        pass

    def _pyro_factor(self, *args, **kwargs):
        """
        Abstract entry point for factor
        """
        pass

    def _pyro_map_data(self, *args, **kwargs):
        """
        Abstract entry point for map_data
        """
        pass

    def _pyro_param(self, *args, **kwargs):
        """
        Abstract entry point for param
        """
        pass

    @property
    def _all_functions(self):
        """
        List of all the primitives this Poutine overrides
        TODO proper docs
        """
        return [
            "sample",
            "observe",
            "factor",
            # "on_exit",
            "param", # XXX ignore for now?
            "map_data"
        ]

