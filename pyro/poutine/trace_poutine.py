import pyro
import torch
from torch.autograd import Variable

from .poutine import Poutine
from pyro.poutine import Trace


class ScalePoutine(Poutine):
    """
    score-rescaling Poutine
    Subsampling means we have to rescale pdfs inside map_data
    This poutine handles the rescaling because it wouldn't fit in Poutine
    """
    def __init__(self, fn, scale):
        """
        Constructor
        """
        self.scale = scale
        super(ScalePoutine, self).__init__(fn)
        self.transparent = False

    def _pyro_sample(self, prev_val, name, fn, *args, **kwargs):
        """
        Rescale the scorer of the stochastic function passed to sample
        """
        pyro.util.rescale_dist_inplace(fn, self.scale)
        return super(ScalePoutine, self)._pyro_sample(
            prev_val, name, fn, *args, **kwargs)

    def _pyro_observe(self, prev_val, name, fn, obs, *args, **kwargs):
        """
        Rescale the scorer of the stochastic function passed to observe
        """
        pyro.util.rescale_dist_inplace(fn, self.scale)
        return super(ScalePoutine, self)._pyro_observe(
            prev_val, name, fn, obs, *args, **kwargs)


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

    def _pyro_sample(self, prev_val, name, dist, *args, **kwargs):
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

        val = super(TracePoutine, self)._pyro_sample(prev_val, name, dist,
                                                     *args, **kwargs)
        self.trace.add_sample(name, val, dist, *args, **kwargs)
        return val

    def _pyro_observe(self, prev_val, name, fn, obs, *args, **kwargs):
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

    def _pyro_map_data(self, prev_val, name, data, fn, batch_size=None):
        """
        Trace map_data
        """
        scale, ind, ind_data = self.get_scale(fn, data, batch_size)
        scaled_fn = ScalePoutine(fn, scale)
        ret = super(TracePoutine, self)._pyro_map_data(prev_val, name,
                                                       data, scaled_fn,
                                                       # XXX watch out for changing
                                                       batch_size=batch_size)
        # store the indices, batch_size, and scaled function in a site
        # XXX does not store input or output values due to space constraints - beware!
        assert hasattr(scaled_fn, "__map_data_indices"), "fn has no __map_data_indices?"
        assert hasattr(scaled_fn, "__map_data_scale"), "fn has no __map_data_scale?"
        if not hasattr(fn, "__map_data_indices"):
            setattr(fn, "__map_data_indices", scaled_fn.__map_data_indices)
            setattr(fn, "__map_data_scale", scaled_fn.__map_data_scale)

        self.trace.add_map_data(name, fn, batch_size,
                                getattr(fn, "__map_data_scale"),
                                getattr(fn, "__map_data_indices"))
        return ret
