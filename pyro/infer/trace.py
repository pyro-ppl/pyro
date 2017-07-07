import pyro


def get_parents(node, trace):
    """
    Get the parents of a node in a trace
    TODO docs
    """
    raise NotImplementedError("not implemented yet")


class Trace(dict):
    """
    Execution trace data structure
    """
    def add_sample(self, name, sample, fn, *args, **kwargs):
        """
        Sample site
        """
        assert(name not in self)
        site = dict({})
        site["type"] = "sample"
        site["value"] = sample
        site["fn"] = fn
        site["args"] = (args, kwargs)
        self[name] = site

    def add_observe(self, name, val, fn, obs, *args, **kwargs):
        """
        Observe site
        """
        assert(name not in self)
        site = dict({})
        site["type"] = "observe"
        site["value"] = val
        site["fn"] = fn
        site["obs"] = obs
        site["args"] = (args, kwargs)
        self[name] = site

    def add_map_data(self, name, data, fn):
        """
        map_data site
        """
        assert(name not in self)
        site = dict({})
        site["type"] = "map_data"
        # XXX
        self[name] = site

    def add_param(self, name, val, *args, **kwargs):
        """
        param site
        """
        site = dict({})
        site["type"] = "param"
        site["value"] = val
        site["args"] = (args, kwargs)
        self[name] = site

    def add_args(self, args_and_kwargs):
        """
        input arguments site
        """
        name = "_INPUT"
        assert(name not in self)
        site = dict({})
        site["type"] = "args"
        site["args"] = args_and_kwargs
        self[name] = site

    def add_return(self, val, *args, **kwargs):
        """
        return value site
        """
        name = "_RETURN"
        assert(name not in self)
        site = dict({})
        site["type"] = "return"
        site["value"] = val
        self[name] = site

    def copy(self):
        """
        Make a copy (for dynamic programming)
        """
        return Trace(self)
