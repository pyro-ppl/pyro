import pyro
import six
from collections import OrderedDict
import torch
from torch.autograd import Variable
from torch.multiprocessing import Pool

from .poutine import Poutine, TagPoutine

def get_parents(site, trace):
    """
    Given a trace and a site name, compute the single-trace Markov blanket of site
    TODO arguments
    TODO formal docs
    """
    # currently, traces are extremely conservative for lack of better implementation
    # a site must be assumed to depend on all previous sites...
    sites = []
    for other_site in trace.keys(): # XXX beware py2/3 compatibility
        if other_site == site:
            break
        sites.append(other_site)
    return sites

class Site(object):
    """
    A single site in an execution trace
    """
    def __init__(self, *args, **kwargs):
        """
        Constructor
        TODO docs
        TODO args
        TODO example
        """
        # XXX not actually implemented yet
        raise NotImplementedError()

class Trace(object):
    """
    Execution trace data structure
    
    TracePoutine creates these
    """
    def __init__(self, uid, *args, **kwargs):
        """
        Constructor
        TODO get right args
        TODO docs
        TODO example
        """
        # XXX not actually implemented yet
        raise NotImplementedError()

class TracePoutine(TagPoutine):
    """
    Execution trace poutine.

    A TracePoutine records the input and output to every pyro primitive
    and stores them as a Site() in a Trace().
    This should, in theory, be sufficient information for every inference algorithm
    (along with the implicit computational graph in the Variables?)

    We can also use this for visualization.
    """

    def __init__(self, fn, sites=None, *args, **kwargs):
        super(TracePoutine, self).__init__(fn, *args, **kwargs)
        self.trace = OrderedDict({})
        self.sites = sites

    def tag_name(self, trace_uid):
        return str(id(self)) + "_{}".format(trace_uid)

    def __call__(self, *args, **kwargs):
        self.trace = OrderedDict({})
        return super(TracePoutine, self).__call__(*args, **kwargs)

    def replay(self, guide_trace, sites=None, *args, **kwargs):
        """
        Replay
        TODO docs
        """
        # check that guide_trace is OrderedDict
        assert(isinstance(guide_trace, OrderedDict))  # XXX add msg
        self.flush_traces()
        # if sites is one name, grab all guide trace sites up to name
        # (in general, all sites that are parents of name)
        if sites is None:
            sites = guide_trace.keys()
            
        if isinstance(sites, str):
            _sites = []
            for k in guide_trace:
                _sites.append(k)
                if k == sites:
                    break
            sites = _sites

        self.guide_trace = guide_trace
        self.sites = sites

        ret = self.__call__(*args, **kwargs)
        #self.guide_trace = OrderedDict({})
        return ret

    def flush_traces(self):
        """
        Convenience function to empty both traces.

        Takes no arguments.
        """
        self.guide_trace = OrderedDict({})
        self.trace = OrderedDict({})

    def _pyro_map_data(self, data, fn, name=None, batch_size=1):
        """
        Trace map_data

        Expected behavior:
        Global batch case:
        if name is None, set name to 'map_data' and fail if that name exists

        If self.guide_trace is empty at that name
        """
        assert(not isinstance(data, torch.Tensor), "only iterator supported")
        if name is None:
            name = "default_map_data"
        assert(name not in self.trace, "name must be unique")
        self.trace[name] = {}
        self.trace[name]["type"] = "map_data"

        # case 1: batch at name not in guide_trace already
        if name not in self.guide_trace:
            batch_ratio = len(data) / batch_size
            # get some random indices
            
        # case 2: batch at name in guide_trace
        else:
            # if it exists, name must correspond to a map_data
            assert(self.guide_trace[name]["type"] == "map_data")
        
        return map(lambda x: fn(x[0], x[1]), self.batch)
        
    def _pyro_sample(self, name, dist, *args, **kwargs):
        """
        sample
        TODO docs
        
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
        ####################################################
        # XXX need to get this behavior implemented properly
        ####################################################
        raise NotImplementedError("CURRENTLY BROKEN")

        if name in self.sites:
            # if the name is in self.sites, proceed as usual
            # make sure the site name is unique
            assert(name not in self.trace)  # XXX add error msg
            self.trace[name] = dict({})
            self.trace[name]["type"] = "sample"  # XXX beware of python2/3 bugs
            self.trace[name]["dist"] = dist
            self.trace[name]["args"] = (args, kwargs)
            self.trace[name]["parents"] = get_parents(name, self.trace)
            # if the site is in the guided subtrace
            if (name in self.guide_trace) and \
            (name in self.sites):
                self.trace[name]["value"] = self.guide_trace[name]["value"]
            else:
                val = dist(*args, **kwargs)
                if dist.reparametrized or not isinstance(val, Variable):
                    self.trace[name]["value"] = val
                else:
                    self.trace[name]["value"] = Variable(val.data)
            return self.trace[name]["value"]
        else:
            # if the name isnt in self.sites, dont record it in the trace
            # and treat it as if it were in the parent trace
            self._pop_stack()
            # XXX this first branch isnt correct
            if name in self.guide_trace.sites:
                val = pyro.sample(name, self.guide_trace[name]["dist"],
                                  *self.guide_trace[name]["args"][0],
                                  **self.guide_trace[name]["args"][1])
            # this one is right
            else:
                val = pyro.sample(name, self.trace[name]["dist"],
                                  *self.trace[name]["args"][0],
                                  **self.trace[name]["args"][1])
            self._push_stack()
            return val

    def _pyro_observe(self, name, dist, val, *args, **kwargs):
        """
        observe
        TODO docs
        
        Expected behavior:
        TODO
        """
        # make sure the site name is unique
        assert(name not in self.trace)  # XXX add error msg
        # for now, we fail if val is None instead of returning a sample
        assert(not (val is None))  # XXX add error msg

        self.trace[name] = dict({})
        self.trace[name]["type"] = "observe"  # XXX beware of py2/3 bugs
        self.trace[name]["dist"] = dist
        self.trace[name]["args"] = (args, kwargs)
        self.trace[name]["parents"] = get_parents(name, self.trace)
        self.trace[name]["value"] = val
        return val

    def _pyro_param(self, name, *args, **kwargs):
        """
        param
        TODO docs
        
        Expected behavior:
        TODO
        """
        # XXX what is correct behavior here
        assert(name not in self.trace)
        # XXX do we want to be able to share params across model and guide?
        #assert(name not in self.guide_trace)
        self.trace[name] = {}
        self.trace[name]["type"] = "param"
        self.trace[name]["args"] = (args, kwargs)
        retrieved = super(TracePoutine, self)._pyro_param(name, *args, **kwargs)
        self.trace[name]["value"] = retrieved
        return retrieved


       
