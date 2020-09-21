# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from functools import partial

from .runtime import _PYRO_STACK


def _context_wrap(context, fn, *args, **kwargs):
    with context:
        return fn(*args, **kwargs)


class _bound_partial(partial):
    """
    Converts a (possibly) bound method into a partial function to
    support class methods as arguments to handlers.
    """
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return partial(self.func, instance)


class Messenger:
    """
    Context manager class that modifies behavior
    and adds side effects to stochastic functions
    i.e. callables containing Pyro primitive statements.

    This is the base Messenger class.
    It implements the default behavior for all Pyro primitives,
    so that the joint distribution induced by a stochastic function fn
    is identical to the joint distribution induced by ``Messenger()(fn)``.

    Class of transformers for messages passed during inference.
    Most inference operations are implemented in subclasses of this.
    """

    def __init__(self):
        pass

    def __call__(self, fn):
        if not callable(fn):
            raise ValueError(
                "{} is not callable, did you mean to pass it as a keyword arg?".format(fn))
        wraps = _bound_partial(partial(_context_wrap, self, fn))
        return wraps

    def __enter__(self):
        """
        :returns: self
        :rtype: pyro.poutine.Messenger

        Installs this messenger at the bottom of the Pyro stack.

        Can be overloaded to add any additional per-call setup functionality,
        but the derived class must always push itself onto the stack, usually
        by calling super().__enter__().

        Derived versions cannot be overridden to take arguments
        and must always return self.
        """
        if not (self in _PYRO_STACK):
            # if this poutine is not already installed,
            # put it on the bottom of the stack.
            _PYRO_STACK.append(self)

            # necessary to return self because the return value of __enter__
            # is bound to VAR in with EXPR as VAR.
            return self
        else:
            # note: currently we raise an error if trying to install a poutine twice.
            # However, this isn't strictly necessary,
            # and blocks recursive poutine execution patterns like
            # like calling self.__call__ inside of self.__call__
            # or with Handler(...) as p: with p: <BLOCK>
            # It's hard to imagine use cases for this pattern,
            # but it could in principle be enabled...
            raise ValueError("cannot install a Messenger instance twice")

    def __exit__(self, exc_type, exc_value, traceback):
        """
        :param exc_type: exception type, e.g. ValueError
        :param exc_value: exception instance?
        :param traceback: traceback for exception handling
        :returns: None
        :rtype: None

        Removes this messenger from the bottom of the Pyro stack.
        If an exception is raised, removes this messenger and everything below it.
        Always called after every execution of self.fn via self.__call__.

        Can be overloaded by derived classes to add any other per-call teardown functionality,
        but the stack must always be popped by the derived class,
        usually by calling super().__exit__(*args).

        Derived versions cannot be overridden to take other arguments,
        and must always return None or False.

        The arguments are the mandatory arguments used by a with statement.
        Users should never be specifying these.
        They are all None unless the body of the with statement raised an exception.
        """
        if exc_type is None:  # callee or enclosed block returned successfully
            # if the callee or enclosed block returned successfully,
            # this poutine should be on the bottom of the stack.
            # If so, remove it from the stack.
            # if not, raise a ValueError because something really weird happened.
            if _PYRO_STACK[-1] == self:
                _PYRO_STACK.pop()
            else:
                # should never get here, but just in case...
                raise ValueError("This Messenger is not on the bottom of the stack")
        else:  # the wrapped function or block raised an exception
            # poutine exception handling:
            # when the callee or enclosed block raises an exception,
            # find this poutine's position in the stack,
            # then remove it and everything below it in the stack.
            if self in _PYRO_STACK:
                loc = _PYRO_STACK.index(self)
                for i in range(loc, len(_PYRO_STACK)):
                    _PYRO_STACK.pop()

    def _reset(self):
        pass

    def _process_message(self, msg):
        """
        :param msg: current message at a trace site
        :returns: None

        Process the message by calling appropriate method of itself based
        on message type. The message is updated in place.
        """
        method = getattr(self, "_pyro_{}".format(msg["type"]), None)
        if method is not None:
            return method(msg)
        return None

    def _postprocess_message(self, msg):
        method = getattr(self, "_pyro_post_{}".format(msg["type"]), None)
        if method is not None:
            return method(msg)
        return None

    @classmethod
    def register(cls, fn=None, type=None, post=None):
        """
        :param fn: function implementing operation
        :param str type: name of the operation
            (also passed to :func:`~pyro.poutine.runtime.effectful`)
        :param bool post: if `True`, use this operation as postprocess

        Dynamically add operations to an effect.
        Useful for generating wrappers for libraries.

        Example::

            @SomeMessengerClass.register
            def some_function(msg)
                ...do_something...
                return msg

        """
        if fn is None:
            return lambda x: cls.register(x, type=type, post=post)

        if type is None:
            raise ValueError("An operation type name must be provided")

        setattr(cls, "_pyro_" + ("post_" if post else "") + type, staticmethod(fn))
        return fn

    @classmethod
    def unregister(cls, fn=None, type=None):
        """
        :param fn: function implementing operation
        :param str type: name of the operation
            (also passed to :func:`~pyro.poutine.runtime.effectful`)

        Dynamically remove operations from an effect.
        Useful for removing wrappers from libraries.

        Example::

            SomeMessengerClass.unregister(some_function, "name")
        """
        if type is None:
            raise ValueError("An operation type name must be provided")

        try:
            delattr(cls, "_pyro_post_" + type)
        except AttributeError:
            pass

        try:
            delattr(cls, "_pyro_" + type)
        except AttributeError:
            pass

        return fn


@contextmanager
def block_messengers(predicate):
    """
    EXPERIMENTAL Context manager to temporarily remove matching messengers from
    the _PYRO_STACK. Note this does not call the ``.__exit__()`` and
    ``.__enter__()`` methods.

    This is useful to selectively block enclosing handlers.

    :param callable predicate: A predicate mapping messenger instance to boolean.
        This mutes all messengers ``m`` for which ``bool(predicate(m)) is True``.
    :yields: A list of matched messengers that are blocked.
    """
    blocked = {}
    try:
        for i, messenger in enumerate(_PYRO_STACK):
            if predicate(messenger):
                blocked[i] = messenger
                _PYRO_STACK[i] = Messenger()  # trivial messenger
        yield list(blocked.values())
    finally:
        for i, messenger in blocked.items():
            _PYRO_STACK[i] = messenger
