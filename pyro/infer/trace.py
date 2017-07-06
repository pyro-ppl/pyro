import pyro


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
        # XXX
        self[name] = site


    def add_observe(self, name, val, fn, obs, *args, **kwargs):
        """
        Observe site
        """
        assert(name not in self)
        site = dict({})
        site["type"] = "observe"
        # XXX
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
        pass


    def add_args(self, args_and_kwargs):
        """
        input arguments site
        """
        name = "_INPUT"
        site = dict({})
        site["type"] = "args"
        # XXX
        self[name] = site


    def add_return(self, val, *args, **kwargs):
        """
        return value site
        """
        name = "_RETURN"
        site = dict({})
        site["type"] = "return"
        # XXX
        self[name] = site
