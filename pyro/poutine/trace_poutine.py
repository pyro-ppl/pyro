import pyro
import torch
from torch.autograd import Variable

from .poutine import Poutine
from pyro.poutine import Trace


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

    def _pyro_map_data(self, prev_val, name, data, fn, batch_size=None, **kwargs):
        """
        Trace map_data
        """
        if batch_size is None:
            batch_size = 0
        assert batch_size >= 0, "cannot have negative batch sizes"

        if isinstance(data, (torch.Tensor, Variable)):  # XXX and np.ndarray?
            if batch_size > 0:
                scale = float(data.size(0)) / float(batch_size)
                ind = Variable(torch.randperm(data.size(0))[0:batch_size])
                ind_data = data.index_select(0, ind)
            else:
                # if batch_size == 0, don't index (saves time/space)
                scale = 1.0
                ind = Variable(torch.range(data.size(0)))
                ind_data = data
            scaled_fn = poutine.scale(fn, scale=scale)
            ret = scaled_fn(ind, ind_data)
        else:
            # if batch_size > 0, select a random set of indices and store it
            if batch_size > 0:
                ind = torch.randperm(len(data))[0:batch_size].numpy().tolist()
                scale = float(len(data)) / float(batch_size)
            else:
                ind = list(xrange(len(data)))
                scale = 1.0
            # map the function over the iterables of indices and data
            scaled_fn = poutine.scale(fn, scale=scale)
            ret = list(map(lambda ix: scaled_fn(*ix), [(i, data[i]) for i in ind]))

        # store the indices, batch_size, and scaled function in a site
        # XXX does not store input or output values due to space constraints - beware!
        self.trace.add_map_data(name, fn, scale, batch_size, ind, **kwargs)
        return ret
