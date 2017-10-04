import torch
from torch.autograd import Variable

import pyro


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
        A sketch of poutine behavior after rewriting as a context manager
        """
        with self:
            return self.orig_fct(*args, **kwargs)

    def __enter__(self):
        """
        Installs this poutine at the bottom of the Pyro stack.
        """
        if not (self in pyro._PYRO_STACK):
            pyro._PYRO_STACK.insert(0, self)
            return self
        else:
            raise ValueError("cannot install a Poutine instance twice")

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Removes this poutine from the bottom of the Pyro stack.
        If an exception is raised, removes this poutine and everything below it.
        """
        if exc_type is None:
            if pyro._PYRO_STACK[0] == self:
                pyro._PYRO_STACK.pop(0)
            else:
                # should never get here, but just in case...
                raise ValueError("This Poutine is not on top of the stack")
        else:
            if self in pyro._PYRO_STACK:
                loc = pyro._PYRO_STACK.index(self)
                for i in range(0, loc + 1):
                    pyro._PYRO_STACK.pop(0)

    # TODO change to more descriptive name that reflects functional asymmetry of up and down
    # possibly gather?
    def down(self, msg):
        """
        :param dict msg: current message at a trace site
        :returns: the updated message at the same trace site

        Adds any information to the message that poutines below it on the stack
        may need to execute properly.

        By default, does nothing, but overridden in derived classes.
        """
        return msg

    def _pyro_sample(self, msg):
        """
        Default pyro.sample Poutine behavior
        """
        fn, args, kwargs = \
            msg["fn"], msg["args"], msg["kwargs"]
        if msg["done"]:
            return msg["ret"]
        val = fn(*args, **kwargs)
        # msg["done"] = True
        return val

    def _pyro_observe(self, msg):
        """
        Default pyro.observe Poutine behavior
        """
        fn, obs, args, kwargs = \
            msg["fn"], msg["obs"], msg["args"], msg["kwargs"]
        if msg["done"]:
            return msg["ret"]
        if obs is None:
            return fn(*args, **kwargs)
        msg["done"] = True
        return obs

    def _pyro_map_data(self, msg):
        """
        Default pyro.map_data Poutine behavior
        """
        data, fn, batch_size, batch_dim = \
            msg["data"], msg["fn"], msg["batch_size"], msg["batch_dim"]
        # we dont want fn to get executed more than once,
        # because then the primitive statements in it will appear multiple times
        if msg["done"]:
            return msg["ret"]
        else:
            if batch_size is None:
                batch_size = 0
            assert batch_size >= 0, "cannot have negative batch sizes"
            if msg["indices"] is None:
                ind = pyro.util.get_batch_indices(data, batch_size, batch_dim)
                msg["indices"] = ind

            if batch_size == 0:
                ind_data = data
            elif isinstance(data, (torch.Tensor, Variable)):  # XXX and np.ndarray?
                ind_data = data.index_select(batch_dim, msg["indices"])
            else:
                ind_data = [data[i] for i in msg["indices"]]

            if isinstance(data, (torch.Tensor, Variable)):
                ret = fn(msg["indices"], ind_data)
            else:
                ret = list(map(lambda ix: fn(*ix), zip(msg["indices"], ind_data)))

            # make sure fn doesn't get reexecuted further up the stack
            msg["done"] = True
            return ret

    def _pyro_param(self, msg):
        """
        overload pyro.param call
        """
        name, args, kwargs = \
            msg["name"], msg["args"], msg["kwargs"]
        if msg["done"]:
            return msg["ret"]
        msg["done"] = True
        return pyro._param_store.get_param(name, *args, **kwargs)
