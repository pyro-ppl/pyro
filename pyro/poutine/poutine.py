import pyro
import torch
from uuid import uuid4 as uuid

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
        # push the current stack onto the pyro global fcts
        self._push_stack()
        self._enter_poutine(*args, **kwargs)

        # run the original function overloading the fcts
        r_val = self.orig_fct(*args, **kwargs)

        # then return the pyro global fcts to their previous state
        self._exit_poutine(r_val, *args, **kwargs)
        self._pop_stack()

        # send back the final val
        return r_val


    def _enter_poutine(self, *args, **kwargs):
        """
        A setup function called right after entry to the Poutine
        """
        self.trace_uid = uuid().hex

    def _exit_poutine(self, r_val, *args, **kwargs):
        """
        A teardown function called right before exit from the Poutine
        """
        pass

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
        """
        loc = pyro._PYRO_STACK.index(self._dispatch)
        for i in range(0, loc+1):
            pyro._PYRO_STACK.pop(0)


    def _tag_name(self, trace_uid):
        """
        Generate a name for the trace consisting of a unique name for the poutine
        and a trace-specific identifier
        """
        return str(id(self)) + "_{}".format(trace_uid)


    def _pyro_sample(self, prev_val, name, fn, *args, **kwargs):
        """
        Default pyro.sample Poutine behavior
        """
        val = fn(*args, **kwargs)
        if self.transparent and not (prev_val is None):
            return prev_val
        else:
            return val

    def _pyro_observe(self, prev_val, name, fn, obs, *args, **kwargs):
        """
        Default pyro.observe Poutine behavior
        """
        if self.transparent and not (prev_val is None):
            return prev_val
        else:
            if obs is None:
                return fn(*args, **kwargs)
            else:
                return obs


    def _pyro_map_data(self, prev_val, name, data, fn):
        """
        Default pyro.map_data Poutine behavior
        """
        if self.transparent and not (prev_val is None):
            return prev_val
        else:
            if isinstance(data, torch.Tensor):
                # assume vectorized observation fn
                raise NotImplementedError(
                    "map_data for vectorized data not yet implemented.")
            else:
                # note that fn should expect an index and a datum
                map(fn, enumerate(data))


    def _pyro_param(self, prev_val, name, *args, **kwargs):
        """
        overload pyro.param call
        here we tag all parameters constructed during this with
        guide. This does not bin them in different buckets, but marks
        for later filtering by inference alg
        """
        return pyro._param_store.get_param(
            tag=self._tag_name(self.trace_uid), *args, **kwargs)


    def get_last_trace_parameters(self):
        """
        grab only the parameters that were called inside of the guide
        e.g. pyro.param(name, val) was called from within guide fct
        """
        return pyro._param_store.filter_parameters(
            self._tag_name(self.trace_uid))

