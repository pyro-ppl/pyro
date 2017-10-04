import pyro

from .poutine import Poutine
from .lambda_poutine import LambdaPoutine
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
        self.trace = Trace()
        self.trace.add_args((args, kwargs))
        ret = super(TracePoutine, self).__call__(*args, **kwargs)
        self.trace.add_return(ret)
        return ret

    def get_trace(self, *args, **kwargs):
        """
        :returns: data structure
        :rtype: pyro.poutine.Trace

        Calls this poutine and returns its trace instead of the function's return value.
        """
        self(*args, **kwargs)
        return self.trace

    def _pyro_sample(self, msg):
        """
        sample
        TODO docs
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
        observe
        TODO docs
        Expected behavior:
        TODO
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

    def _pyro_param(self, msg):
        """
        param
        TODO docs
        Expected behavior:
        TODO
        """
        name, args, kwargs = \
            msg["name"], msg["args"], msg["kwargs"]
        retrieved = super(TracePoutine, self)._pyro_param(msg)
        self.trace.add_param(name, retrieved, *args, **kwargs)
        return retrieved

    def _pyro_map_data(self, msg):
        """
        Trace map_data
        """
        name, data, fn, batch_size, batch_dim = \
            msg["name"], msg["data"], msg["fn"], msg["batch_size"], msg["batch_dim"]

        scale = pyro.util.get_batch_scale(data, batch_size, batch_dim)
        msg.update({"fn": LambdaPoutine(fn, name, scale)})

        ret = super(TracePoutine, self)._pyro_map_data(msg)
        self.trace.add_map_data(name, fn, batch_size, batch_dim, msg["indices"])
        return ret
