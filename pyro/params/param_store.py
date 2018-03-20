from __future__ import absolute_import, division, print_function

import torch


class ParamStoreDict(object):
    """
    Global store for parameters in Pyro. This is basically a key-value store.
    The typical user interacts with the ParamStore primarily through the
    primitive `pyro.param`.

    See `Intro Part II <http://pyro.ai/examples/intro_part_ii.html>`_ for further discussion
    and `SVI Part I <http://pyro.ai/examples/svi_part_i.html>`_ for some examples.

    Some things to bear in mind when using parameters in Pyro:

    - parameters must be assigned unique names
    - the `init_tensor` argument to `pyro.param` is only used the first time that a given (named)
      parameter is registered with Pyro.
    - for this reason, a user may need to use the `clear()` method if working in a REPL in order to
      get the desired behavior. this method can also be invoked with `pyro.clear_param_store()`.
    - the internal name of a parameter within a PyTorch `nn.Module` that has been registered with
      Pyro is prepended with the Pyro name of the module. so nothing prevents the user from having
      two different modules each of which contains a parameter named `weight`. by contrast, a user
      can only have one top-level parameter named `weight` (outside of any module).
    - parameters can be saved and loaded from disk using `save` and `load`.
    """

    def __init__(self):
        """
        initialize ParamStore data structures
        """
        self._params = {}  # dictionary from param name to param
        self._param_to_name = {}  # dictionary from param to param name
        self._active_params = set()  # set of all currently active params

    def clear(self):
        """
        Clear the ParamStore
        """
        self._params = {}
        self._param_to_name = {}
        self._active_params = set()

    def named_parameters(self):
        """
        Returns an iterator over tuples of the form (name, parameter) for each parameter in the ParamStore
        """
        return self._params.items()

    def get_all_param_names(self):
        """
        Get all parameter names in the ParamStore
        """
        return self._params.keys()

    def get_active_params(self, tags=None):
        """
        :returns: all active params in the ParamStore
        :rtype: set
        """
        assert tags is None, "removing tag support"
        return self._active_params

    def mark_params_active(self, params):
        """
        :param params: iterable of params the user wishes to mark as active in the ParamStore.
            this information is used to determine which parameters are being optimized,
            e.g. in the context of pyro.infer.SVI
        """
        assert(all([p in self._param_to_name for p in params])), \
            "some of these parameters are not in the ParamStore"
        self._active_params.update(set(params))

    def mark_params_inactive(self, params):
        """
        :param params: iterable of params the user wishes to mark as inactive in the ParamStore.
            this information is used to determine which parameters are being optimized,
            e.g. in the context of pyro.infer.SVI
        """
        assert(all([p in self._param_to_name for p in params])), \
            "some of these parameters are not in the ParamStore"
        self._active_params.difference_update(set(params))

    def replace_param(self, param_name, new_param, old_param):
        """
        Replace the param param_name with current value old_param with the new value new_param

        :param param_name: parameter name
        :type param_name: str
        :param new_param: the paramater to be put into the ParamStore
        :type new_param: torch.Tensor
        :param old_param: the paramater to be removed from the ParamStore
        :type new_param: torch.Tensor
        """
        assert id(self._params[param_name]) == id(old_param)
        self._params[param_name] = new_param
        self._param_to_name[new_param] = param_name
        self._param_to_name.pop(old_param)

    def get_param(self, name, init_tensor=None, tags=None):
        """
        Get parameter from its name. If it does not yet exist in the
        ParamStore, it will be created and stored.
        The Pyro primitive `pyro.param` dispatches to this method.

        :param name: parameter name
        :type name: str
        :param init_tensor: initial tensor
        :type init_tensor: torch.Tensor
        :returns: parameter
        :rtype: torch.Tensor
        """
        assert tags is None, "removing tag support"
        if name not in self._params:
            # if not create the init tensor through
            assert init_tensor is not None,\
                "cannot initialize a parameter '{}' with None. Did you get the param name right?".format(name)

            # a function
            if callable(init_tensor):
                self._params[name] = init_tensor()
            else:
                # from the memory passed in
                self._params[name] = init_tensor

            # keep track of each tensor and it's name
            self._param_to_name[self._params[name]] = name

        # send back the guaranteed to exist param
        return self._params[name]

    def param_name(self, p):
        """
        Get parameter name from parameter

        :param p: parameter
        :returns: parameter name
        """
        if p not in self._param_to_name:
            return None

        return self._param_to_name[p]

    def get_state(self):
        """
        Get the ParamStore state.
        """
        state = (self._params,)
        return state

    def set_state(self, state):
        """
        Set the ParamStore state using state from a previous get_state() call
        """
        assert isinstance(state, tuple) and len(state) == 1, "malformed ParamStore state"
        loaded_params, = state

        for param_name, param in loaded_params.items():
            self._params[param_name] = param
            self._param_to_name[param] = param_name

    def save(self, filename):
        """
        Save parameters to disk

        :param filename: file name to save to
        :type name: str
        """
        with open(filename, "wb") as output_file:
            torch.save(self.get_state(), output_file)

    def load(self, filename):
        """
        Loads parameters from disk

        :param filename: file name to load from
        :type name: str
        """
        with open(filename, "rb") as input_file:
            state = torch.load(input_file)
        self.set_state(state)
