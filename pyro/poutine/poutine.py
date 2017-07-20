import pyro
import torch
from torch.autograd import Variable

from . import scale as poutine_scale


class Poutine(object):
    """
    Wraps a function call with a pyro stack push/pop of the basic pyro functions
    This is the base class with all APIs implemented and sane default behavior.
    """
    def __init__(self, fn):
        """
        Constructor
        """
        # store original fn to wrap
        self.orig_fct = fn
        self.transparent = True

    def __call__(self, *args, **kwargs):
        """
        Wrap the original function call to call the poutine object
        """
        try:
            # push the current stack onto the pyro global fcts
            self._push_stack()
            self._enter_poutine(*args, **kwargs)

            # run the original function overloading the fcts
            base_r_val = self.orig_fct(*args, **kwargs)

            # then return the pyro global fcts to their previous state
            r_val = self._exit_poutine(base_r_val, *args, **kwargs)
            self._pop_stack()

            # send back the final val
            return r_val
        except Exception as e:
            self._flush_stack()
            raise

    def _enter_poutine(self, *args, **kwargs):
        """
        A setup function called right after entry to the Poutine
        """
        pass

    def _exit_poutine(self, r_val, *args, **kwargs):
        """
        A teardown function called right before exit from the Poutine
        """
        return r_val

    def _dispatch(self, site_type, _ret, name, *args, **kwargs):
        """
        The dispatcher that gets put into _PYRO_STACK
        """
        ret = getattr(self, "_pyro_" + site_type)(_ret, name, *args, **kwargs)
        barrier = self._block_stack(site_type, name)
        return ret, barrier

    def _block_stack(self, site_type, name):
        """
        Default behavior for stack-blocking:
        In general, don't stop operating the stack at that site
        """
        return False

    def _push_stack(self):
        """
        Store the current stack of pyro functions, push this class model fcts
        """
        if not (self._dispatch in pyro._PYRO_STACK):
            pyro._PYRO_STACK.insert(0, self._dispatch)
        else:
            raise ValueError("cannot install a Poutine instance twice")

    def _pop_stack(self):
        """
        Reset global pyro attributes to the previously recorded fcts
        """
        if pyro._PYRO_STACK[0] == self._dispatch:
            pyro._PYRO_STACK.pop(0)
        else:
            raise ValueError("This Poutine is not on top of the stack")

    def _flush_stack(self):
        """
        Find our dispatcher in the stack, then remove it and everything below it
        Needed for exception handling
        """
        if self._dispatch in pyro._PYRO_STACK:
            loc = pyro._PYRO_STACK.index(self._dispatch)
            for i in range(0, loc + 1):
                pyro._PYRO_STACK.pop(0)

    def _pyro_sample(self, prev_val, name, fn, *args, **kwargs):
        """
        Default pyro.sample Poutine behavior
        """
        if self.transparent and prev_val is not None:
            return prev_val
        val = fn(*args, **kwargs)
        return val

    def _pyro_observe(self, prev_val, name, fn, obs, *args, **kwargs):
        """
        Default pyro.observe Poutine behavior
        """
        if self.transparent and not (prev_val is None):
            return prev_val
        if obs is None:
            return fn(*args, **kwargs)
        return obs

    def _pyro_map_data(self, prev_val, name, data, fn, batch_size=None, **kwargs):
        """
        Default pyro.map_data Poutine behavior
        """
        if self.transparent and not (prev_val is None):
            return prev_val
        else:
            if batch_size is None:
                batch_size = 0
            assert batch_size >= 0, "cannot have negative batch sizes"
            if hasattr(fn, "__map_data_indices") and \
               hasattr(fn, "__map_data_scale"):
                ind = fn.__map_data_indices
                scale = fn.__map_data_scale

            if isinstance(data, (torch.Tensor, Variable)):  # XXX and np.ndarray?
                if batch_size > 0:
                    if not hasattr(fn, "__map_data_indices"):
                        scale = float(data.size(0)) / float(batch_size)
                        ind = Variable(torch.randperm(data.size(0))[0:batch_size])
                    ind_data = data.index_select(0, ind)
                else:
                    # if batch_size == 0, don't index (saves time/space)
                    scale = 1.0
                    ind = Variable(torch.range(data.size(0)))
                    ind_data = data
                scaled_fn = poutine_scale(fn, scale=scale)
                ret = scaled_fn(ind, ind_data)
            else:
                # if batch_size > 0, select a random set of indices and store it
                if batch_size > 0 and not hasattr(fn, "__map_data_indices"):
                    ind = torch.randperm(len(data))[0:batch_size].numpy().tolist()
                    scale = float(len(data)) / float(batch_size)
                else:
                    ind = list(range(len(data)))
                    scale = 1.0
                # map the function over the iterables of indices and data
                scaled_fn = poutine_scale(fn, scale=scale)
                ret = list(map(lambda ix: scaled_fn(*ix), [(i, data[i]) for i in ind]))
            # XXX is there a more elegant way to move indices up the stack?
            if not hasattr(fn, "__map_data_indices"):
                setattr(fn, "__map_data_indices", ind)
                setattr(fn, "__map_data_scale", scale)
            return ret

    def _pyro_param(self, prev_val, name, *args, **kwargs):
        """
        overload pyro.param call
        """
        if self.transparent and prev_val is not None:
            return prev_val
        return pyro._param_store.get_param(name, *args, **kwargs)
