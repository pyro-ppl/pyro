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
        return self.trace

    def _pyro_sample(self, msg, name, dist, *args, **kwargs):
        """
        sample
        TODO docs
        """
        if name in self.trace:
            # XXX temporary solution - right now, if the name appears in the trace,
            # we assume that this was intentional and that the poutine restarted,
            # so we should reset self.trace to be empty
            self._enter_poutine(*self.trace["_INPUT"]["args"][0],
                                **self.trace["_INPUT"]["args"][1])

        val = super(TracePoutine, self)._pyro_sample(msg, name, dist,
                                                     *args, **kwargs)
        self.trace.add_sample(name, msg["scale"], val, dist, *args, **kwargs)
        return val

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        """
        observe
        TODO docs
        Expected behavior:
        TODO
        """
        if name in self.trace:
            # XXX temporary solution - right now, if the name appears in the trace,
            # we assume that this was intentional and that the poutine restarted,
            # so we should reset self.trace to be empty
            self._enter_poutine(*self.trace["_INPUT"]["args"][0],
                                **self.trace["_INPUT"]["args"][1])

        val = super(TracePoutine, self)._pyro_observe(msg, name, fn, obs,
                                                      *args, **kwargs)
        self.trace.add_observe(name, msg["scale"], val, fn, obs, *args, **kwargs)
        return val

    def _pyro_param(self, msg, name, *args, **kwargs):
        """
        param
        TODO docs
        Expected behavior:
        TODO
        """
        retrieved = super(TracePoutine, self)._pyro_param(msg, name,
                                                          *args, **kwargs)
        self.trace.add_param(name, retrieved, *args, **kwargs)
        return retrieved

    def _pyro_map_data(self, msg, name, data, fn, batch_size=None, batch_dim=0):
        """
        Trace map_data
        """
        scale = pyro.util.get_batch_scale(data, batch_size, batch_dim)
        ret = super(TracePoutine, self)._pyro_map_data(msg, name, data,
                                                       LambdaPoutine(fn, name, scale),
                                                       # XXX watch out for changing
                                                       batch_size=batch_size,
                                                       batch_dim=batch_dim)

        self.trace.add_map_data(name, fn, batch_size, batch_dim, msg["indices"])
        return ret
