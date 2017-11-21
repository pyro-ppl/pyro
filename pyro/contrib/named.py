"""
Named Data Structures
---------------------

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
    >>> mu = state.mu.param_(Variable(torch.zeros(1), requires_grad=True))
    >>> sigma = state.sigma.param_(Variable(torch.ones(1), requires_grad=True))
    >>> z = state.z.sample_(dist.normal, mu, sigma)
    >>> state.x.observe_(dist.normal, z, mu, sigma)

For deeper examples of how these can be used in model code, see the
`Tree Data <https://github.com/uber/pyro/blob/dev/examples/contrib/named/tree_data.py>`_
and
`Mixture <https://github.com/uber/pyro/blob/dev/examples/contrib/named/mixture.py>`_
examples.

Authors: Fritz Obermeyer, Alexander Rush
"""

from __future__ import absolute_import, division, print_function

import contextlib
import functools
from inspect import isclass

import pyro
import pyro.util


def _is_named(obj):
    return isinstance(obj, (Object, List, Dict))


class Object(object):
    """
    Object to hold immutable latent state.

    This object can serve either as a container for nested latent state
    or as a placeholder to be replaced by a Variable via a named.sample,
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
    def __init__(self, name="latent"):
        super(Object, self).__setattr__("_name", name)
        super(Object, self).__setattr__("_is_placeholder", True)

    def _local_items(self):
        return ((k, v) for k, v in self.__dict__.items()
                if k[0] != "_")

    def _items(self):
        for key, val in self._local_items():
            if _is_named(val):
                for k, v in val._items():
                    yield k, v
            else:
                name = "{}.{}".format(self._name, key)
                yield name, val

    def __str__(self):
        return self._name

    def __repr__(self):
        out = "Object (\n"
        for k, v in self._local_items():
            out += "{} : {}\n".format(k, repr(v))
        out += ")"
        return out

    def __getattribute__(self, key):
        try:
            return super(Object, self).__getattribute__(key)
        except AttributeError:
            name = "{}.{}".format(self._name, key)
            value = Object(name)
            super(Object, value).__setattr__(
                "_set_value", lambda value: super(Object, self).__setattr__(key, value))
            super(Object, self).__setattr__(key, value)
            super(Object, self).__setattr__("_is_placeholder", False)
            return value

    def __setattr__(self, key, value):
        if isinstance(value, (List, Dict)):
            value._set_name("{}.{}".format(self._name, key))
        if hasattr(self, key):
            old = super(Object, self).__getattribute__(key)
            if not isinstance(old, Object) or not old._is_placeholder:
                raise RuntimeError("Cannot overwrite {}.{}".format(self._name, key))
        super(Object, self).__setattr__(key, value)

    @functools.wraps(pyro.sample)
    def sample_(self, fn, *args, **kwargs):
        if not self._is_placeholder:
            raise RuntimeError("Cannot .sample_ an initialized named.Object {}".format(self._name))
        value = pyro.sample(self._name, fn, *args, **kwargs)
        self._set_value(value)
        return value

    @functools.wraps(pyro.observe)
    def observe_(self, fn, obs, *args, **kwargs):
        if not self._is_placeholder:
            raise RuntimeError("Cannot .observe_ an initialized named.Object {}".format(self._name))
        value = pyro.observe(self._name, fn, obs, *args, **kwargs)
        self._set_value(value)
        return value

    @functools.wraps(pyro.param)
    def param_(self, *args, **kwargs):
        if not self._is_placeholder:
            raise RuntimeError("Cannot .param_ an initialized named.Object")
        value = pyro.param(self._name, *args, **kwargs)
        self._set_value(value)
        return value

    def set_(self, value):
        if not self._is_placeholder:
            raise RuntimeError("Cannot .set_ an initialized named.Object")
        self._set_value(value)
        return value

    @contextlib.contextmanager
    @functools.wraps(pyro.iarange)
    def iarange_(self, *args, **kwargs):
        if not self._is_placeholder:
            raise RuntimeError("Cannot .iarange_ an initialized named.Object")

        # Yields both a subsampled data and an indexed latent object.
        with pyro.iarange(self._name, *args, **kwargs) as ind:
            yield ind

    @functools.wraps(pyro.irange)
    def irange_(self, *args, **kwargs):
        return pyro.irange(self._name, *args, **kwargs)  # Returns an iterator.

    def ienumerate_(self, ls, *args, **kwargs):
        self.plate = List()
        for i in self.irange.irange_(len(ls), *args, **kwargs):
            yield i, ls[i], self.plate.add()

    @functools.wraps(pyro.module)
    def module_(self, nn_module, tags="default"):
        if not self._is_placeholder:
            raise RuntimeError("Cannot .module_ an initialized named.Object")

        assert hasattr(nn_module, "parameters"), "module has no parameters"
        if isclass(nn_module):
            raise NotImplementedError("pyro.module does not support class constructors for " +
                                      "the argument nn_module")

        for param_name, param_value in nn_module.named_parameters():
            # register the parameter in the module with pyro
            # this only does something substantive if the parameter hasn't been seen before
            pyro.util.deep_getattr(self, param_name).param_(param_value, tags=tags)
        return nn_module


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
    def __init__(self, name=None, size=None):
        self._name = name
        self._size = size

    def _set_name(self, name):
        if self:
            raise RuntimeError("Cannot name a named.List after data has been added")
        if self._name is not None:
            raise RuntimeError("Cannot rename named.List: {}".format(self._name))
        self._name = name
        if self._size is not None:
            for i in range(self._size):
                self.add()

    def _items(self):
        for i, obj in enumerate(self):
            if _is_named(obj):
                for k, v in obj._items():
                    yield k, v
            else:
                name = "{}[{}]".format(self._name, i)
                yield name, obj

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

    def __repr__(self):
        return "List ( {} )".format(", ".join(map(repr, self)))

    def __str__(self):
        return self._name

    def __setitem__(self, pos, value):
        name = "{}[{}]".format(self._name, pos)
        if isinstance(value, Object):
            raise RuntimeError("Cannot store named.Object {} in named.Dict {}".format(value, self._name))
        elif isinstance(value, (List, Dict)):
            value._set_name(name)
        old = self[pos]
        if not isinstance(old, Object) or not old._is_placeholder:
            raise RuntimeError("Cannot overwrite {}".format(name))
        super(List, self).__setitem__(pos, value)


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

    def _set_name(self, name):
        if self:
            raise RuntimeError("Cannot name a named.Dict after data has been added")
        if self._name is not None:
            raise RuntimeError("Cannot rename named.Dict: {}".format(self._name))
        self._name = name

    def _items(self):
        for key, val in self.items():
            if _is_named(val):
                for k, v in val._items():
                    yield k, v
            else:
                name = "{}[{!r}]".format(self._name, key)
                yield name, val

    def __repr__(self):
        return "Dict (\n {} \n)".format(",".join(("{}: {}".format(k, repr(v))
                                                  for k, v in self.items())))

    def __str__(self):
        return self._name

    def __getitem__(self, key):
        try:
            return super(Dict, self).__getitem__(key)
        except KeyError:
            if self._name is None:
                raise RuntimeError("Cannot access an unnamed named.Dict")
            value = Object("{}[{!r}]".format(self._name, key))
            super(Object, value).__setattr__(
                "_set_value", lambda value: self.__setitem__(key, value))
            super(Dict, self).__setitem__(key, value)
            return value

    def __setitem__(self, key, value):
        name = "{}[{!r}]".format(self._name, key)
        if key in self:
            old = super(Dict, self).__getitem__(key)
            if not isinstance(old, Object) or not old._is_placeholder:
                raise RuntimeError("Cannot overwrite {}".format(name))
        if isinstance(value, Object):
            raise RuntimeError("Cannot store named.Object {} in named.Dict {}".format(value, self._name))
        elif isinstance(value, (List, Dict)):
            value._set_name(name)
        super(Dict, self).__setitem__(key, value)
