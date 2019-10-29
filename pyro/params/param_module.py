import weakref
from collections import OrderedDict

import torch
from .constrained_parameter import ConstrainedParameter


class StaticConstraint:
    def __init__(self, name, constraint):
        assert isinstance(constraint, torch.distributions.constraints.Constraint)
        self.__name__ = name
        self.constraint = constraint

    def __call__(self, obj):
        return self.constraint


# This is a singleton object.
class ParamStoreModule(torch.nn.Module):
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
        self._constraints = OrderedDict()
        super().__init__()

    def add_constraint(self, name, constraint):
        """
        Adds a constraint for a parameter.
        """
        if hasattr(self, name):
            raise AttributeError("Cannot constrain an existing parameter: {}".format(name))
        if hasattr(ParamStoreModule, name):
            raise AttributeError("Cannot overwrite constraint for {}".format(name))
        constraint_fn = StaticConstraint(name, constraint)
        descriptor = ConstrainedParameter(name, constraint_fn)
        setattr(ParamStoreModule, name, descriptor)

    def __delattr__(self, name):
        if name in self._constraints:
            del self._constraints[name]
            del ParamStoreModule.__dict__[name]
            delattr(self, name + "_unconstrained")

    def _clear_constraints(self):
        while self._constraints:
            delattr(self, self._constraints.pop())

    def __setstate__(self, state):
        super().__setstate__(state)

        # Update class attributes from singleton instance's ._constraints.
        self._clear_constraints()
        for name, constraint in self._constraints.items():
            self._add_constraint(name, constraint)

    # -------------------------------------------------------------------------------
    # Old dict-like interface

    def clear(self):
        """
        Clear the ParamStore.
        """
        self._parameters.clear()
        self._buffers.clear()
        self._modules.clear()
        self._clear_constraints()

    def items(self):
        """
        Iterate over ``(name, constrained_param)`` pairs.
        """
        for name in self._constraints:
            yield name, self[name]

    def keys(self):
        """
        Iterate over param names.
        """
        return self._constraints.keys()

    def values(self):
        """
        Iterate over constrained parameter values.
        """
        for name in self._constraints:
            yield self[name]

    def __bool__(self):
        return bool(self._constraints)

    def __len__(self):
        return len(self._constraints)

    def __contains__(self, name):
        return name in self._constraints

    def __iter__(self):
        """
        Iterate over param names.
        """
        return iter(self._constraints)

    def __delitem__(self, name):
        """
        Remove a parameter from the param store.
        """
        del self._constraints[name]
        del ParamStoreModule.__dict__[name]
        delattr(self, name + "_unconstrained")

    def __getitem__(self, name):
        """
        Get the constrained value of a named parameter.
        """
        unconstrained_value = getattr(self, name + "_unconstrained")
        constrained_value = getattr(self, name)
        constrained_value.unconstrained = weakref.ref(unconstrained_value)
        return constrained_value

    def __setitem__(self, name, new_constrained_value):
        """
        Set the constrained value of an existing parameter, or the value of a
        new unconstrained parameter. To declare a new parameter with
        constraint, use :meth:`setdefault`.
        """
        if name not in self._constraints:
            self._add_constraint(name, constraints.positive)
        setattr(self, name, new_constrained_value)

    def setdefault(self, name, init_constrained_value, constraint=constraints.real):
        """
        Retrieve a constrained parameter value from the if it exists, otherwise
        set the initial value. Note that this is a little fancier than
        :meth:`dict.setdefault`.

        If the parameter already exists, ``init_constrained_tensor`` will be ignored. To avoid
        expensive creation of ``init_constrained_tensor`` you can wrap it in a ``lambda`` that
        will only be evaluated if the parameter does not already exist::

            param_store.get("foo", lambda: (0.001 * torch.randn(1000, 1000)).exp(),
                            constraint=constraints.positive)

        :param str name: parameter name
        :param init_constrained_value: initial constrained value
        :type init_constrained_value: torch.Tensor or callable returning a torch.Tensor
        :param constraint: torch constraint object
        :type constraint: torch.distributions.constraints.Constraint
        :returns: constrained parameter value
        :rtype: torch.Tensor
        """
        if name not in self._constraints:
            # set the constraint
            self._add_constraint(name, constraint)

            # evaluate the lazy value
            if callable(init_constrained_value):
                init_constrained_value = init_constrained_value()

            # set the initial value
            setattr(self, name, init_constrained_value)

        # get the param, which is guaranteed to exist
        return self[name]
