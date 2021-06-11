# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
The ``pyro.contrib.named`` module is a thin syntactic layer on top of Pyro.  It
allows Pyro models to be written to look like programs with operating on Python
data structures like ``latent.x.sample_(...)``, rather than programs with
string-labeled statements like ``x = pyro.sample("x", ...)``.

This module provides three container data structures ``named.Object``,
``named.List``, and ``named.Dict``. These data structures are intended to be
nested in each other. Together they track the address of each piece of data
in each data structure, so that this address can be used as a Pyro site. For
example::

    >>> state = named.Object("state")
    >>> print(str(state))
    state

    >>> z = state.x.y.z  # z is just a placeholder.
    >>> print(str(z))
    state.x.y.z

    >>> state.xs = named.List()  # Create a contained list.
    >>> x0 = state.xs.add()
    >>> print(str(x0))
    state.xs[0]

    >>> state.ys = named.Dict()
    >>> foo = state.ys['foo']
    >>> print(str(foo))
    state.ys['foo']

These addresses can now be used inside ``sample``, ``observe`` and ``param``
statements. These named data structures even provide in-place methods that
alias Pyro statements. For example::

    >>> state = named.Object("state")
    >>> loc = state.loc.param_(torch.zeros(1, requires_grad=True))
    >>> scale = state.scale.param_(torch.ones(1, requires_grad=True))
    >>> z = state.z.sample_(dist.Normal(loc, scale))
    >>> obs = state.x.sample_(dist.Normal(loc, scale), obs=z)

For deeper examples of how these can be used in model code, see the
`Tree Data <https://github.com/pyro-ppl/pyro/blob/dev/examples/contrib/named/tree_data.py>`_
and
`Mixture <https://github.com/pyro-ppl/pyro/blob/dev/examples/contrib/named/mixture.py>`_
examples.

Authors: Fritz Obermeyer, Alexander Rush
"""
import functools

import pyro


class Object:
    """
    Object to hold immutable latent state.

    This object can serve either as a container for nested latent state
    or as a placeholder to be replaced by a tensor via a named.sample,
    named.observe, or named.param statement. When used as a placeholder,
    Object objects take the place of strings in normal pyro.sample statements.

    :param str name: The name of the object.

    Example::

        state = named.Object("state")
        state.x = 0
        state.ys = named.List()
        state.zs = named.Dict()
        state.a.b.c.d.e.f.g = 0  # Creates a chain of named.Objects.

    .. warning:: This data structure is write-once: data may be added but may
        not be mutated or removed. Trying to mutate this data structure may
        result in silent errors.
    """
    def __init__(self, name):
        super().__setattr__("_name", name)
        super().__setattr__("_is_placeholder", True)

    def __str__(self):
        return super().__getattribute__("_name")

    def __getattribute__(self, key):
        try:
            return super().__getattribute__(key)
        except AttributeError:
            name = "{}.{}".format(self, key)
            value = Object(name)
            super(Object, value).__setattr__(
                "_set_value", lambda value: super(Object, self).__setattr__(key, value))
            super().__setattr__(key, value)
            super().__setattr__("_is_placeholder", False)
            return value

    def __setattr__(self, key, value):
        if isinstance(value, (List, Dict)):
            value._set_name("{}.{}".format(self, key))
        if hasattr(self, key):
            old = super().__getattribute__(key)
            if not isinstance(old, Object) or not old._is_placeholder:
                raise RuntimeError("Cannot overwrite {}.{}".format(self, key))
        super().__setattr__(key, value)

    @functools.wraps(pyro.sample)
    def sample_(self, fn, *args, **kwargs):
        if not self._is_placeholder:
            raise RuntimeError("Cannot .sample_ an initialized named.Object {}".format(self))
        value = pyro.sample(str(self), fn, *args, **kwargs)
        self._set_value(value)
        return value

    @functools.wraps(pyro.param)
    def param_(self, *args, **kwargs):
        if not self._is_placeholder:
            raise RuntimeError("Cannot .param_ an initialized named.Object")
        value = pyro.param(str(self), *args, **kwargs)
        self._set_value(value)
        return value


class List(list):
    """
    List-like object to hold immutable latent state.

    This must either be given a name when constructed::

        latent = named.List("root")

    or must be immediately stored in a ``named.Object``::

        latent = named.Object("root")
        latent.xs = named.List()  # Must be bound to a Object before use.

    .. warning:: This data structure is write-once: data may be added but may
        not be mutated or removed. Trying to mutate this data structure may
        result in silent errors.
    """
    def __init__(self, name=None):
        self._name = name

    def __str__(self):
        return self._name

    def _set_name(self, name):
        if self:
            raise RuntimeError("Cannot name a named.List after data has been added")
        if self._name is not None:
            raise RuntimeError("Cannot rename named.List: {}".format(self._name))
        self._name = name

    def add(self):
        """
        Append one new named.Object.

        :returns: a new latent object at the end
        :rtype: named.Object
        """
        if self._name is None:
            raise RuntimeError("Cannot .add() to a named.List before storing it in a named.Object")
        i = len(self)
        value = Object("{}[{}]".format(self._name, i))
        super(Object, value).__setattr__(
            "_set_value", lambda value, i=i: self.__setitem__(i, value))
        self.append(value)
        return value

    def __setitem__(self, pos, value):
        name = "{}[{}]".format(self._name, pos)
        if isinstance(value, Object):
            raise RuntimeError("Cannot store named.Object {} in named.Dict {}".format(value, self._name))
        elif isinstance(value, (List, Dict)):
            value._set_name(name)
        old = self[pos]
        if not isinstance(old, Object) or not old._is_placeholder:
            raise RuntimeError("Cannot overwrite {}".format(name))
        super().__setitem__(pos, value)


class Dict(dict):
    """
    Dict-like object to hold immutable latent state.

    This must either be given a name when constructed::

        latent = named.Dict("root")

    or must be immediately stored in a ``named.Object``::

        latent = named.Object("root")
        latent.xs = named.Dict()  # Must be bound to a Object before use.

    .. warning:: This data structure is write-once: data may be added but may
        not be mutated or removed. Trying to mutate this data structure may
        result in silent errors.
    """
    def __init__(self, name=None):
        self._name = name

    def __str__(self):
        return self._name

    def _set_name(self, name):
        if self:
            raise RuntimeError("Cannot name a named.Dict after data has been added")
        if self._name is not None:
            raise RuntimeError("Cannot rename named.Dict: {}".format(self._name))
        self._name = name

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            if self._name is None:
                raise RuntimeError("Cannot access an unnamed named.Dict") from e
            value = Object("{}[{!r}]".format(self._name, key))
            super(Object, value).__setattr__(
                "_set_value", lambda value: self.__setitem__(key, value))
            super().__setitem__(key, value)
            return value

    def __setitem__(self, key, value):
        name = "{}[{!r}]".format(self._name, key)
        if key in self:
            old = super().__getitem__(key)
            if not isinstance(old, Object) or not old._is_placeholder:
                raise RuntimeError("Cannot overwrite {}".format(name))
        if isinstance(value, Object):
            raise RuntimeError("Cannot store named.Object {} in named.Dict {}".format(value, self._name))
        elif isinstance(value, (List, Dict)):
            value._set_name(name)
        super().__setitem__(key, value)
