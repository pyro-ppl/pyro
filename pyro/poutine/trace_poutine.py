from .poutine import Poutine
from .trace import Trace


class TracePoutine(Poutine):
    """
    Execution trace poutine.

    A TracePoutine records the input and output to every pyro primitive
    and stores them as a site in a Trace().
    This should, in theory, be sufficient information for every inference algorithm
    (along with the implicit computational graph in the Variables?)

    We can also use this for visualization.
    """

    def __init__(self, fn, graph_type=None):
        """
        TODO docs
        """
        if graph_type is None:
            graph_type = "flat"
        assert graph_type in ("flat", "dense")
        self.graph_type = graph_type
        super(TracePoutine, self).__init__(fn)

    def __call__(self, *args, **kwargs):
        """
        Runs the stochastic function stored in this poutine,
        with additional side effects.

        Resets self.trace to an empty trace,
        installs itself on the global execution stack,
        runs self.fn with the given arguments,
        uninstalls itself from the global execution stack,
        stores the arguments and return value of the function in special sites,
        and returns self.fn's return value
        """
        self.trace = Trace(graph_type=self.graph_type)
        self.trace.add_node("_INPUT",
                            name="_INPUT", type="args",
                            args=args, kwargs=kwargs)
        ret = super(TracePoutine, self).__call__(*args, **kwargs)
        self.trace.add_node("_RETURN", name="_RETURN", type="return", value=ret)
        self.trace.identify_edges()
        return ret

    def get_trace(self, *args, **kwargs):
        """
        :returns: data structure
        :rtype: pyro.poutine.Trace

        Helper method for a very common use case.
        Calls this poutine and returns its trace instead of the function's return value.
        """
        self(*args, **kwargs)
        return self.trace.copy()

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.

        Implements default pyro.sample Poutine behavior with a side effect
        call the function at the site,
        store the result in self.trace,
        and return the result
        """
        if msg["name"] in self.trace:
            # XXX temporary solution - right now, if the name appears in the trace,
            # we assume that this was intentional and that the poutine restarted,
            # so we should reset self.trace to be empty
            tr = Trace(graph_type=self.graph_type)
            tr.add_node("_INPUT",
                        name="_INPUT", type="input",
                        args=self.trace.nodes["_INPUT"]["args"],
                        kwargs=self.trace.nodes["_INPUT"]["kwargs"])
            self.trace = tr

        val = super(TracePoutine, self)._pyro_sample(msg)
        site = msg.copy()
        site.update(value=val)
        self.trace.add_node(msg["name"], **site)
        return val

    def _pyro_observe(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: the observed value at the site.

        Implements default pyro.observe Poutine behavior with an additional side effect:
        if the observation at the site is not None,
        then store the observation in self.trace
        and return the observation,
        else call the function,
        then store the return value in self.trace
        and return the return value.
        """
        if msg["name"] in self.trace:
            # XXX temporary solution - right now, if the name appears in the trace,
            # we assume that this was intentional and that the poutine restarted,
            # so we should reset self.trace to be empty
            tr = Trace(graph_type=self.graph_type)
            tr.add_node("_INPUT",
                        name="_INPUT", type="input",
                        args=self.trace.nodes["_INPUT"]["args"],
                        kwargs=self.trace.nodes["_INPUT"]["kwargs"])
            self.trace = tr

        val = super(TracePoutine, self)._pyro_observe(msg)
        site = msg.copy()
        site.update(value=val)
        self.trace.add_node(msg["name"], **site)
        return val

    def _pyro_param(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: the result of querying the parameter store

        Implements default pyro.param Poutine behavior with an additional side effect:
        queries the parameter store with the site name and varargs
        and returns the result of the query.

        If the parameter doesn't exist, create it using the site varargs.
        If it does exist, grab it from the parameter store.
        Store the parameter in self.trace, and then return the parameter.
        """
        val = super(TracePoutine, self)._pyro_param(msg)
        site = msg.copy()
        site.update(value=val)
        self.trace.add_node(msg["name"], **site)
        return val
