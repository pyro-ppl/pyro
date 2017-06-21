import pyro
import torch
from uuid import uuid4 as uuid


class Poutine(object):
    """
    Wraps a function call with a pyro stack push/pop of the basic pyro functions
    """
    # default functions
    _all_functions = [
        "sample",
        "observe",
        # "on_exit",
        "param",
        "map_data"
    ]

    def __init__(self, fn):
        # store original fn to wrap
        self.orig_fct = fn

    def _enter_poutine(self, *args, **kwargs):
        pass

    def _exit_poutine(self, *args, **kwargs):
        pass

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
        self._exit_poutine(*args, **kwargs)
        self._pop_stack()

        # send back the final val
        return r_val

    @staticmethod
    def _get_current_stack():
        return {af: getattr(pyro, af) for af in Poutine._all_functions}

    # save current state
    def _push_stack(self):
        '''
        Store the current stack of pyro functions, push this class model fcts
        '''

        # previous save stack
        self._previous_stack = Poutine._get_current_stack()

        # replace current pyro functions
        for af in Poutine._all_functions:
            if hasattr(self, "_pyro_" + af):
                setattr(pyro, af, getattr(self, "_pyro_" + af))

    def _pop_stack(self):
        '''
        Reset global pyro attributes to the previously recorded fcts
        '''

        # loop through the previous stack dictionary of function calls
        for af in self._previous_stack:
            # set the global to be the stack obj
            setattr(pyro, af, self._previous_stack[af])

        # erase previous stack
        self._previous_stack = None

    # All the pyro calls
    def _pyro_sample(self, *args, **kwargs):
        raise NotImplementedError(
            "Abstract Infer class does not implement this function")

    def _pyro_observe(self, *args, **kwargs):
        raise NotImplementedError(
            "Abstract Infer class does not implement this function")

    # def _pyro_on_exit(self, *args, **kwargs):
    #   raise NotImplementedError("Abstract Infer class does not implement this function")

    # def _pyro_param(self, *args, **kwargs):
    #   raise NotImplementedError("Abstract Infer class does not implement this function")

    def _pyro_map_data(self, data, fn):
        if isinstance(data, torch.Tensor):
            # assume vectorized observation fn
            raise NotImplementedError(
                "map_data for vectorized data not yet implemented.")
        else:
            # note that fn should expect an index and a datum
            map(fn, enumerate(data))


class TagPoutine(Poutine):
    """
    This class overrides the default poutine to include overriding
    pyro.param. It tags every pyro.param call with a custom string.
    """

    def __init__(self, *args, **kwargs):
        super(TagPoutine, self).__init__(*args, **kwargs)
        self.trace_uid = 0

    def _enter_poutine(self, *args, **kwargs):
        """
        When entering the poutine for the first time, update our trace id
        """
        self.trace_uid = uuid().hex

    def tag_name(self, trace_uid):
        raise NotImplementedError(
            "Must implement unique tagging of poutines in TagPoutine class")

    # overload pyro.param call
    # here we tag all parameters constructed during this with
    # guide. This does not bin them in different buckets, but marks
    # for later filtering by inference alg
    def _pyro_param(self, *args, **kwargs):
        return pyro._param_store.get_param(
            tag=self.tag_name(self.trace_uid), *args, **kwargs)

    # grab only the parameters that were called inside of the guide
    # e.g. pyro.param(name, val) was called from within guide fct
    def get_last_trace_parameters(self):
        return pyro._param_store.filter_parameters(
            self.tag_name(self.trace_uid))
