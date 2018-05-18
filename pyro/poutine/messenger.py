from __future__ import absolute_import, division, print_function

from .runtime import _PYRO_STACK


class Messenger(object):
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
        def _wraps(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        _wraps.msngr = self
        return _wraps

    def __enter__(self):
        """
        :returns: self
        :rtype: pyro.poutine.Messenger

        Installs this messenger at the bottom of the Pyro stack.

        Can be overloaded to add any additional per-call setup functionality,
        but the derived class must always push itself onto the stack, usually
        by calling super(Derived, self).__enter__().

        Derived versions cannot be overridden to take arguments
        and must always return self.
        """
        if not (self in _PYRO_STACK):
            # if this poutine is not already installed,
            # put it on the bottom of the stack.
            _PYRO_STACK.insert(0, self)

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
        usually by calling super(Derived, self).__exit__(*args).

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
            if _PYRO_STACK[0] == self:
                _PYRO_STACK.pop(0)
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
                for i in range(0, loc + 1):
                    _PYRO_STACK.pop(0)

    def _reset(self):
        pass

    def _process_message(self, msg):
        """
        :param msg: current message at a trace site
        :returns: None

        Process the message by calling appropriate method of itself based
        on message type. The message is updated in place.
        """
        return getattr(self, "_pyro_{}".format(msg["type"]))(msg)

    def _postprocess_message(self, msg):
        return None

    def _pyro_sample(self, msg):
        return None

    def _pyro_param(self, msg):
        return None
