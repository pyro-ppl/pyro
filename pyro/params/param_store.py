from __future__ import absolute_import, division, print_function

import weakref

import torch
from torch.distributions import constraints, transform_to


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
        self._param_to_name = {}  # dictionary from unconstrained param to param name
        self._constraints = {}  # dictionary from param name to constraint object

    def clear(self):
        """
        Clear the ParamStore
        """
        self._params = {}
        self._param_to_name = {}
        self._constraints = {}

    def named_parameters(self):
        """
        Returns an iterator over tuples of the form (name, parameter) for each parameter in the ParamStore
        """
        # TODO consider returing constrained
        return self._params.items()

    def get_all_param_names(self):
        """
        Get all parameter names in the ParamStore
        """
        return self._params.keys()

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
        assert self._params[param_name] is old_param.unconstrained()
        del self._params[param_name]
        del self._param_to_name[old_param.unconstrained()]
        self.get_param(param_name, new_param, constraint=self._constraints[param_name])

    def get_param(self, name, init_tensor=None, constraint=constraints.real):
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
        if name not in self._params:
            # if not create the init tensor through
            assert init_tensor is not None,\
                "cannot initialize a parameter '{}' with None. Did you get the param name right?".format(name)

            # a function
            if callable(init_tensor):
                init_tensor = init_tensor()

            # store the unconstrained value and constraint
            with torch.no_grad():
                unconstrained_param = transform_to(constraint).inv(init_tensor)
            unconstrained_param.requires_grad_(True)
            self._params[name] = unconstrained_param
            self._constraints[name] = constraint

            # keep track of each tensor and it's name
            self._param_to_name[unconstrained_param] = name

        elif init_tensor is not None and not callable(init_tensor):
            if self._params[name].shape != init_tensor.shape:
                raise ValueError("param {} init tensor shape does not match existing value: {} vs {}".format(
                    name, init_tensor.shape, self._params[name].shape))

        # get the guaranteed to exist param
        unconstrained_param = self._params[name]

        # compute the constrained value
        param = transform_to(self._constraints[name])(unconstrained_param)
        param.unconstrained = weakref.ref(unconstrained_param)

        return param

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
        state = {
            'params': self._params,
            'constraints': self._constraints,
        }
        return state

    def set_state(self, state):
        """
        Set the ParamStore state using state from a previous get_state() call
        """
        assert isinstance(state, dict), "malformed ParamStore state"
        assert set(state.keys()) == set(['params', 'constraints']), \
            "malformed ParamStore keys {}".format(state.keys())

        for param_name, param in state['params'].items():
            self._params[param_name] = param
            self._param_to_name[param] = param_name

        for param_name, constraint in state['constraints'].items():
            if isinstance(constraint, type(constraints.real)):
                # Work around lack of hash & equality comparison on constraints.
                constraint = constraints.real
            self._constraints[param_name] = constraint

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


# used to create fully-formed param names, e.g. mymodule$$$mysubmodule.weight
_MODULE_NAMESPACE_DIVIDER = "$$$"


def param_with_module_name(pyro_name, param_name):
    return _MODULE_NAMESPACE_DIVIDER.join([pyro_name, param_name])


def module_from_param_with_module_name(param_name):
    return param_name.split(_MODULE_NAMESPACE_DIVIDER)[0]


def user_param_name(param_name):
    if _MODULE_NAMESPACE_DIVIDER in param_name:
        return param_name.split(_MODULE_NAMESPACE_DIVIDER)[1]
    return param_name
