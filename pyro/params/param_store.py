# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import re
import warnings
import weakref
from contextlib import contextmanager
from typing import (
    Callable,
    Dict,
    ItemsView,
    Iterator,
    KeysView,
    Optional,
    Tuple,
    Union,
)

import torch
from torch.distributions import constraints, transform_to
from torch.serialization import MAP_LOCATION
from typing_extensions import TypedDict


class StateDict(TypedDict):
    params: Dict[str, torch.Tensor]
    constraints: Dict[str, constraints.Constraint]


class ParamStoreDict:
    """
    Global store for parameters in Pyro. This is basically a key-value store.
    The typical user interacts with the ParamStore primarily through the
    primitive `pyro.param`.

    See `Introduction <http://pyro.ai/examples/intro_long.html>`_ for further discussion
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
    - in general parameters are associated with both *constrained* and *unconstrained* values. for
      example, under the hood a parameter that is constrained to be positive is represented as an
      unconstrained tensor in log space.
    """

    # -------------------------------------------------------------------------------
    # New dict-like interface

    def __init__(self) -> None:
        """
        initialize ParamStore data structures
        """
        self._params: Dict[str, torch.Tensor] = (
            {}
        )  # dictionary from param name to param
        self._param_to_name: Dict[torch.Tensor, str] = (
            {}
        )  # dictionary from unconstrained param to param name
        self._constraints: Dict[str, constraints.Constraint] = (
            {}
        )  # dictionary from param name to constraint object

    def clear(self) -> None:
        """
        Clear the ParamStore
        """
        self._params = {}
        self._param_to_name = {}
        self._constraints = {}

    def items(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """
        Iterate over ``(name, constrained_param)`` pairs. Note that `constrained_param` is
        in the constrained (i.e. user-facing) space.
        """
        for name in self._params:
            yield name, self[name]

    def keys(self) -> KeysView[str]:
        """
        Iterate over param names.
        """
        return self._params.keys()

    def values(self) -> Iterator[torch.Tensor]:
        """
        Iterate over constrained parameter values.
        """
        for name, constrained_param in self.items():
            yield constrained_param

    def __bool__(self) -> bool:
        return bool(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __contains__(self, name: str) -> bool:
        return name in self._params

    def __iter__(self) -> Iterator[str]:
        """
        Iterate over param names.
        """
        return iter(self.keys())

    def __delitem__(self, name) -> None:
        """
        Remove a parameter from the param store.
        """
        unconstrained_value = self._params.pop(name)
        self._param_to_name.pop(unconstrained_value)
        self._constraints.pop(name)

    def __getitem__(self, name: str) -> torch.Tensor:
        """
        Get the *constrained* value of a named parameter.
        """
        unconstrained_value = self._params[name]

        # compute the constrained value
        constraint = self._constraints[name]
        constrained_value: torch.Tensor = transform_to(constraint)(unconstrained_value)
        constrained_value.unconstrained = weakref.ref(unconstrained_value)  # type: ignore[attr-defined]

        return constrained_value

    def __setitem__(self, name: str, new_constrained_value: torch.Tensor) -> None:
        """
        Set the constrained value of an existing parameter, or the value of a
        new *unconstrained* parameter. To declare a new parameter with
        constraint, use :meth:`setdefault`.
        """
        # store constraint, defaulting to unconstrained
        constraint = self._constraints.setdefault(name, constraints.real)

        # compute the unconstrained value
        with torch.no_grad():
            # FIXME should we .detach() the new_constrained_value?
            unconstrained_value = transform_to(constraint).inv(new_constrained_value)
            unconstrained_value = unconstrained_value.contiguous()
        unconstrained_value.requires_grad_(True)

        # store a bidirectional mapping between name and unconstrained tensor
        self._params[name] = unconstrained_value
        self._param_to_name[unconstrained_value] = name

    def setdefault(
        self,
        name: str,
        init_constrained_value: Union[torch.Tensor, Callable[[], torch.Tensor]],
        constraint: constraints.Constraint = constraints.real,
    ) -> torch.Tensor:
        """
        Retrieve a *constrained* parameter value from the ``ParamStoreDict`` if it exists,
        otherwise set the initial value. Note that this is a little fancier than
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
        :type constraint: ~torch.distributions.constraints.Constraint
        :returns: constrained parameter value
        :rtype: torch.Tensor
        """
        if name not in self._params:
            # set the constraint
            self._constraints[name] = constraint

            # evaluate the lazy value
            if callable(init_constrained_value):
                init_constrained_value = init_constrained_value()

            # set the initial value
            self[name] = init_constrained_value

        # get the param, which is guaranteed to exist
        return self[name]

    # -------------------------------------------------------------------------------
    # Old non-dict interface

    def named_parameters(self) -> ItemsView[str, torch.Tensor]:
        """
        Returns an iterator over ``(name, unconstrained_value)`` tuples for
        each parameter in the ParamStore. Note that, in the event the parameter is constrained,
        `unconstrained_value` is in the unconstrained space implicitly used by the constraint.
        """
        return self._params.items()

    def get_all_param_names(self) -> KeysView[str]:
        warnings.warn(
            "ParamStore.get_all_param_names() is deprecated; use .keys() instead.",
            DeprecationWarning,
        )
        return self.keys()

    def replace_param(
        self, param_name: str, new_param: torch.Tensor, old_param: torch.Tensor
    ) -> None:
        warnings.warn(
            "ParamStore.replace_param() is deprecated; use .__setitem__() instead.",
            DeprecationWarning,
        )
        assert self._params[param_name] is old_param.unconstrained()  # type: ignore[attr-defined]
        self[param_name] = new_param

    def get_param(
        self,
        name: str,
        init_tensor: Optional[torch.Tensor] = None,
        constraint: constraints.Constraint = constraints.real,
        event_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Get parameter from its name. If it does not yet exist in the
        ParamStore, it will be created and stored.
        The Pyro primitive `pyro.param` dispatches to this method.

        :param name: parameter name
        :type name: str
        :param init_tensor: initial tensor
        :type init_tensor: torch.Tensor
        :param constraint: torch constraint
        :type constraint: torch.distributions.constraints.Constraint
        :param int event_dim: (ignored)
        :returns: parameter
        :rtype: torch.Tensor
        """
        if init_tensor is None:
            return self[name]
        else:
            return self.setdefault(name, init_tensor, constraint)

    def match(self, name: str) -> Dict[str, torch.Tensor]:
        """
        Get all parameters that match regex. The parameter must exist.

        :param name: regular expression
        :type name: str
        :returns: dict with key param name and value torch Tensor
        """
        pattern = re.compile(name)
        return {name: self[name] for name in self if pattern.match(name)}

    def param_name(self, p: torch.Tensor) -> Optional[str]:
        """
        Get parameter name from parameter

        :param p: parameter
        :returns: parameter name
        """
        return self._param_to_name.get(p)

    # -------------------------------------------------------------------------------
    # Persistence interface

    def get_state(self) -> StateDict:
        """
        Get the ParamStore state.
        """
        params = self._params.copy()
        # Remove weakrefs in preparation for pickling.
        for param in params.values():
            param.__dict__.pop("unconstrained", None)
        state: StateDict = {"params": params, "constraints": self._constraints.copy()}
        return state

    def set_state(self, state: StateDict) -> None:
        """
        Set the ParamStore state using state from a previous :meth:`get_state` call
        """
        assert isinstance(state, dict), "malformed ParamStore state"
        assert set(state.keys()) == set(
            ["params", "constraints"]
        ), "malformed ParamStore keys {}".format(state.keys())

        for param_name, param in state["params"].items():
            self._params[param_name] = param
            self._param_to_name[param] = param_name

        for param_name, constraint in state["constraints"].items():
            if isinstance(constraint, type(constraints.real)):
                # Work around lack of hash & equality comparison on constraints.
                constraint = constraints.real
            self._constraints[param_name] = constraint

    def save(self, filename: str) -> None:
        """
        Save parameters to file

        :param filename: file name to save to
        :type filename: str
        """
        with open(filename, "wb") as output_file:
            torch.save(self.get_state(), output_file)

    def load(self, filename: str, map_location: MAP_LOCATION = None) -> None:
        """
        Loads parameters from file

        .. note::

           If using :meth:`pyro.module` on parameters loaded from
           disk, be sure to set the ``update_module_params`` flag::

               pyro.get_param_store().load('saved_params.save')
               pyro.module('module', nn, update_module_params=True)

        :param filename: file name to load from
        :type filename: str
        :param map_location: specifies how to remap storage locations
        :type map_location: function, torch.device, string or a dict
        """
        with open(filename, "rb") as input_file:
            state = torch.load(input_file, map_location, weights_only=False)
        self.set_state(state)

    @contextmanager
    def scope(self, state: Optional[StateDict] = None) -> Iterator[StateDict]:
        """
        Context manager for using multiple parameter stores within the same process.

        This is a thin wrapper around :meth:`get_state`, :meth:`clear`, and
        :meth:`set_state`. For large models where memory space is limiting, you
        may want to instead manually use :meth:`save`, :meth:`clear`, and
        :meth:`load`.

        Example usage::

            param_store = pyro.get_param_store()

            # Train multiple models, while avoiding param name conflicts.
            with param_store.scope() as scope1:
                # ...Train one model,guide pair...
            with param_store.scope() as scope2:
                # ...Train another model,guide pair...

            # Now evaluate each, still avoiding name conflicts.
            with param_store.scope(scope1):  # loads the first model's scope
               # ...evaluate the first model...
            with param_store.scope(scope2):  # loads the second model's scope
               # ...evaluate the second model...
        """
        if state is None:
            state = {"params": {}, "constraints": {}}
        old_state = self.get_state()
        try:
            self.clear()
            self.set_state(state)
            yield state
            state.update(self.get_state())
        finally:
            self.clear()
            self.set_state(old_state)


# used to create fully-formed param names, e.g. mymodule$$$mysubmodule.weight
_MODULE_NAMESPACE_DIVIDER = "$$$"


def param_with_module_name(pyro_name: str, param_name: str) -> str:
    return _MODULE_NAMESPACE_DIVIDER.join([pyro_name, param_name])


def module_from_param_with_module_name(param_name: str) -> str:
    return param_name.split(_MODULE_NAMESPACE_DIVIDER)[0]


def user_param_name(param_name: str) -> str:
    if _MODULE_NAMESPACE_DIVIDER in param_name:
        return param_name.split(_MODULE_NAMESPACE_DIVIDER)[1]
    return param_name


def normalize_param_name(name: str) -> str:
    return name.replace(_MODULE_NAMESPACE_DIVIDER, ".")
