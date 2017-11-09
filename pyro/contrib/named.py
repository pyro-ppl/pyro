from __future__ import absolute_import, division, print_function

import functools

import pyro


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
    def __init__(self, name):
        super(Object, self).__setattr__("_name", name)
        super(Object, self).__setattr__("_is_placeholder", True)

    def __str__(self):
        return super(Object, self).__getattribute__("_name")

    def __getattribute__(self, key):
        try:
            return super(Object, self).__getattribute__(key)
        except AttributeError:
            name = "{}.{}".format(self, key)
            value = Object(name)
            super(Object, value).__setattr__(
                "_set_value", lambda value: super(Object, self).__setattr__(key, value))
            super(Object, self).__setattr__(key, value)
            super(Object, self).__setattr__("_is_placeholder", False)
            return value

    def __setattr__(self, key, value):
        if isinstance(value, (List, Dict)):
            value._set_name("{}.{}".format(self, key))
        if hasattr(self, key):
            old = super(Object, self).__getattribute__(key)
            if not isinstance(old, Object) or not old._is_placeholder:
                raise RuntimeError("Cannot overwrite {}.{}".format(self, key))
        super(Object, self).__setattr__(key, value)

    @functools.wraps(pyro.sample)
    def sample_(self, fn, *args, **kwargs):
        if not self._is_placeholder:
            raise RuntimeError("Cannot .sample_ an initialized named.Object {}".format(self))
        value = pyro.sample(str(self), fn, *args, **kwargs)
        self._set_value(value)
        return value

    @functools.wraps(pyro.observe)
    def observe_(self, fn, obs, *args, **kwargs):
        if not self._is_placeholder:
            raise RuntimeError("Cannot .observe_ an initialized named.Object {}".format(self))
        value = pyro.observe(str(self), fn, obs, *args, **kwargs)
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


@functools.wraps(pyro.sample)
def sample(placeholder, fn, *args, **kwargs):
    if not isinstance(placeholder, Object):
        raise TypeError("named.sample expected a named.Object but got {}".format(repr(placeholder)))
    return placeholder.sample(fn, *args, **kwargs)


@functools.wraps(pyro.observe)
def observe(placeholder, fn, obs, *args, **kwargs):
    if not isinstance(placeholder, Object):
        raise TypeError("named.observe expected a named.Object but got {}".format(repr(placeholder)))
    return placeholder.observe(fn, obs, *args, **kwargs)


@functools.wraps(pyro.param)
def param(placeholder, *args, **kwargs):
    if not isinstance(placeholder, Object):
        return placeholder  # value was already set
    return placeholder.param(*args, **kwargs)
