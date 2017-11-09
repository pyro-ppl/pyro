from __future__ import absolute_import, division, print_function

import functools

import pyro


class Latent(object):
    """
    Object to hold latent state.

    This object can serve either as a container for nested latent state
    or as a placeholder to be replaced by a Variable via a sugar.sample,
    sugar.observe, or sugar.param statement. When used as a placeholder,
    Latent objects take the place of strings in normal pyro.sample statements.

    :param str address: The name of the object.

    Example::

        state = Latent("state")
        state.x = 0
        state.ys = LatentList(5)
        state.zs = LatentDict()
        state.a.b.c.d.e.f.g = 0  # Creates a chain of Latents.
    """
    def __init__(self, address):
        super(Latent, self).__setattr__('_address', address)

    def __str__(self):
        return super(Latent, self).__getattribute__('_address')

    def __getattribute__(self, name):
        try:
            return super(Latent, self).__getattribute__(name)
        except AttributeError:
            address = '{}.{}'.format(self, name)
            value = Latent(address)
            value._set = lambda value: super(Latent, self).__setattr__(name, value)
            super(Latent, self).__setattr__(name, value)
            return value

    def __setattr__(self, name, value):
        if isinstance(value, (LatentList, LatentDict)):
            value._bind('{}.{}'.format(self, name))
        super(Latent, self).__setattr__(name, value)


class LatentList(list):
    """
    List-like object to hold latent state.

    This must be created in expressions like::

        latent = Latent()
        latent.xs = LatentList(5)  # Must be bound to a Latent before use.

    :param int size: The initial size of the list. After the list is created
        and bound to a Latent object, it will be filled with ``size``
        ``Latent`` objects that can be used as placeholders or containers.

    .. warning:: Advanced mutation is not supported and may fail silently.
        For example do not sort, reverse, or delete items from the list.
    """
    def __init__(self, size=0):
        self._size = size
        self._address = None

    def _bind(self, address):
        if self:
            raise RuntimeError("Cannot bind a LatentList in a Latent after data has been added")
        if self._address is not None:
            raise RuntimeError("Tried to bind an already-bound LatentList: {}".format(self._address))
        self._address = address
        for i in range(self._size):
            value = Latent('{}[{}]'.format(address, i))
            value._set = lambda value, i=i: self.__setitem__(i, value)
            self.append(value)

    def add(self):
        """
        Append one new Latent object.

        :returns: a new latent object at the end
        :rtype: Latent
        """
        if self._address is None:
            raise RuntimeError("Cannot .add() to a LatentList before binding it to a Latent")
        i = len(self)
        value = Latent('{}[{}]'.format(self._address, i))
        value._set = lambda value, i=i: self.__setitem__(i, value)
        self.append(value)
        return value


class LatentDict(dict):
    """
    Temporary dict-like object to hold latent state.

    This must be created in expressions like::

        latent = Latent()
        latent.xs = LatentDict()  # Must be bound to a Latent before use.
    """
    def __init__(self):
        self._address = None

    def _bind(self, address):
        if self:
            raise RuntimeError("Cannot bind a LatentDict in a Latent after data has been added")
        if self._address is not None:
            raise RuntimeError("Tried to bind an already-bound LatentDict: {}".format(self._address))
        self._address = address

    def __getitem__(self, key):
        try:
            return super(LatentDict, self).__getitem__(key)
        except KeyError:
            if self._address is None:
                raise RuntimeError("Cannot access a LatentDict until it is bound to a Latent")
            value = Latent('{}[{!r}]'.format(self._address, key))
            value._set = lambda value: self.__setitem__(key, value)
            super(LatentDict, self).__setitem__(key, value)
            return value

    def __setitem__(self, key, value):
        if isinstance(value, Latent):
            raise RuntimeError("Cannot bind Latent {} to LatentDict {}".format(value, self._address))
        super(LatentDict, self).__setitem__(key, value)


@functools.wraps(pyro.sample)
def sample(latent, fn, *args, **kwargs):
    if not isinstance(latent, Latent):
        raise TypeError("sugar.sample expected a Latent but got {}".format(repr(latent)))
    value = pyro.sample(str(latent), fn, *args, **kwargs)
    latent._set(value)
    return value


@functools.wraps(pyro.observe)
def observe(latent, fn, obs, *args, **kwargs):
    if not isinstance(latent, Latent):
        raise TypeError("sugar.observe expected a Latent but got {}".format(repr(latent)))
    value = pyro.observe(str(latent), fn, obs, *args, **kwargs)
    latent._set(value)
    return value


@functools.wraps(pyro.param)
def param(latent, *args, **kwargs):
    if not isinstance(latent, Latent):
        return latent  # value was already set
    value = pyro.param(str(latent), *args, **kwargs)
    latent._set(value)
    return value
