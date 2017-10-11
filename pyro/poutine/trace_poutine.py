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
        self.trace = Trace()
        self.trace.add_args((args, kwargs))
        ret = super(TracePoutine, self).__call__(*args, **kwargs)
        self.trace.add_return(ret)
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
        name, fn, args, kwargs = \
            msg["name"], msg["fn"], msg["args"], msg["kwargs"]
        if name in self.trace:
            # XXX temporary solution - right now, if the name appears in the trace,
            # we assume that this was intentional and that the poutine restarted,
            # so we should reset self.trace to be empty
            tr = Trace()
            tr.add_args(self.trace["_INPUT"]["args"])
            self.trace = tr

        val = super(TracePoutine, self)._pyro_sample(msg)
        self.trace.add_sample(name, msg["scale"], val, fn, *args, **kwargs)
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
        name, fn, obs, args, kwargs = \
            msg["name"], msg["fn"], msg["obs"], msg["args"], msg["kwargs"]
        if name in self.trace:
            # XXX temporary solution - right now, if the name appears in the trace,
            # we assume that this was intentional and that the poutine restarted,
            # so we should reset self.trace to be empty
            tr = Trace()
            tr.add_args(self.trace["_INPUT"]["args"])
            self.trace = tr

        val = super(TracePoutine, self)._pyro_observe(msg)
        self.trace.add_observe(name, msg["scale"], val, fn, obs, *args, **kwargs)
        return val

    def _pyro_managed(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: the result of running a managed computation.

        Implements default pyro.managed Poutine behavior:
        executes a simple function.

        Derived classes often compute a side effect,
        then call super(Derived, self)._pyro_managed(msg).
        """
        name, fn, args, kwargs = \
            msg["name"], msg["fn"], msg["args"], msg["kwargs"]
        retrieved = super(TracePoutine, self)._pyro_managed(msg)
        self.trace.add_managed(name, retrieved, fn, *args, **kwargs)
        return retrieved

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
        name, args, kwargs = \
            msg["name"], msg["args"], msg["kwargs"]
        retrieved = super(TracePoutine, self)._pyro_param(msg)
        self.trace.add_param(name, retrieved, *args, **kwargs)
        return retrieved
