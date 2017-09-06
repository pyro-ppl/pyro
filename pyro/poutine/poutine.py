import pyro
import torch
from torch.autograd import Variable


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

    def _block_up(self, msg):
        """
        Default behavior for stack-blocking:
        In general, don't stop operating the stack at that site
        """
        return False

    def _block_down(self, msg):
        """
        Block going down
        """
        return False

    def up(self, msg):
        """
        The dispatcher that gets put into _PYRO_STACK
        """
        # TODO can probably condense this logic, keeping explicit for now
        if msg["type"] == "sample":
            ret = self._pyro_sample(msg, msg["name"],
                                    msg["fn"],
                                    *msg["args"], **msg["kwargs"])
        elif msg["type"] == "observe":
            ret = self._pyro_observe(msg, msg["name"],
                                     msg["fn"], msg["val"],
                                     *msg["args"], **msg["kwargs"])
        elif msg["type"] == "param":
            ret = self._pyro_param(msg, msg["name"],
                                   *msg["args"], **msg["kwargs"])
        elif msg["type"] == "map_data":
            ret = self._pyro_map_data(msg, msg["name"],
                                      msg["data"], msg["fn"], msg["batch_size"])
        else:
            raise ValueError(
                "{} is an invalid site type, how did that get there?".format(msg["type"]))

        msg.update({"ret": ret})
        barrier = self._block_up(msg)
        return msg, barrier

    def down(self, msg):
        """
        The dispatcher that gets put into _PYRO_STACK
        """
        barrier = self._block_down(msg)
        return msg, barrier

    def _push_stack(self):
        """
        Store the current stack of pyro functions, push this class model fcts
        """
        if not (self in pyro._PYRO_STACK):
            pyro._PYRO_STACK.insert(0, self)
        else:
            raise ValueError("cannot install a Poutine instance twice")

    def _pop_stack(self):
        """
        Reset global pyro attributes to the previously recorded fcts
        """
        if pyro._PYRO_STACK[0] == self:
            pyro._PYRO_STACK.pop(0)
        else:
            raise ValueError("This Poutine is not on top of the stack")

    def _flush_stack(self):
        """
        Find our dispatcher in the stack, then remove it and everything below it
        Needed for exception handling
        """
        if self in pyro._PYRO_STACK:
            loc = pyro._PYRO_STACK.index(self)
            for i in range(0, loc + 1):
                pyro._PYRO_STACK.pop(0)

    def _pyro_sample(self, msg, name, fn, *args, **kwargs):
        """
        Default pyro.sample Poutine behavior
        """
        if msg["ret"] is not None:
            return msg["ret"]
        val = fn(*args, **kwargs)
        return val

    def _pyro_observe(self, msg, name, fn, obs, *args, **kwargs):
        """
        Default pyro.observe Poutine behavior
        """
        if msg["ret"] is not None:
            return msg["ret"]
        if obs is None:
            return fn(*args, **kwargs)
        return obs

    def _pyro_map_data(self, msg, name, data, fn, batch_size):
        """
        Default pyro.map_data Poutine behavior
        """
        if msg["ret"] is not None:
            return msg["ret"]
        else:
            if batch_size is None:
                batch_size = 0
            assert batch_size >= 0, "cannot have negative batch sizes"
            if msg["scale"] is None and msg["indices"] is None:
                scale, ind = pyro.util.get_scale(data, batch_size)
                msg["scale"] = scale
                msg["indices"] = ind

            if batch_size == 0:
                ind_data = data
            elif isinstance(data, (torch.Tensor, Variable)):  # XXX and np.ndarray?
                ind_data = data.index_select(0, msg["indices"])
            else:
                ind_data = [data[i] for i in msg["indices"]]

            if isinstance(data, (torch.Tensor, Variable)):
                ret = fn(msg["indices"], ind_data)
            else:
                ret = list(map(lambda ix: fn(*ix), enumerate(ind_data)))
            return ret

    def _pyro_param(self, msg, name, *args, **kwargs):
        """
        overload pyro.param call
        """
        if msg["ret"] is not None:
            return msg["ret"]
        return pyro._param_store.get_param(name, *args, **kwargs)
