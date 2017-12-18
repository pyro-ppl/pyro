from __future__ import absolute_import, division, print_function

from pyro.params import _PYRO_PARAM_STORE

# the global pyro stack
_PYRO_STACK = []


class Poutine(object):
    """
    Context manager class that modifies behavior
    and adds side effects to stochastic functions
    i.e. callables containing pyro primitive statements.

    See the Poutine execution model writeup in the documentation
    for a description of the entire Poutine system.

    This is the base Poutine class.
    It implements the default behavior for all pyro primitives,
    so that the joint distribution induced by a stochastic function fn
    is identical to the joint distribution induced by Poutine(fn).
    """

    def __init__(self, fn):
        """
        :param fn: a stochastic function (callable containing pyro primitive calls)

        Constructor. Doesn't do much, just stores the stochastic function.
        """
        self.fn = fn

    def __call__(self, *args, **kwargs):
        """
        Installs self onto the global effect stack,
        then calls the stored stochastic function with the given varargs,
        then uninstalls itself from the stack and returns the above value.

        Guaranteed to have the same call signature (input/output type)
        as the stored function.
        """
        with self:
            return self.fn(*args, **kwargs)

    def __enter__(self):
        """
        :returns: self
        :rtype: pyro.poutine.Poutine

        Installs this poutine at the bottom of the Pyro stack.
        Called before every execution of self.fn via self.__call__().

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
            # or with Poutine(...) as p: with p: <BLOCK>
            # It's hard to imagine use cases for this pattern,
            # but it could in principle be enabled...
            raise ValueError("cannot install a Poutine instance twice")

    def __exit__(self, exc_type, exc_value, traceback):
        """
        :param exc_type: exception type, e.g. ValueError
        :param exc_value: exception instance?
        :param traceback: traceback for exception handling
        :returns: None
        :rtype: None

        Removes this poutine from the bottom of the Pyro stack.
        If an exception is raised, removes this poutine and everything below it.
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
                raise ValueError("This Poutine is not on the bottom of the stack")
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
        """
        Resets the computation to the beginning, un-sampling all sample sites.

        By default, does nothing, but overridden in derived classes.
        """
        pass

    def _prepare_site(self, msg):
        """
        :param msg: current message at a trace site
        :returns: the updated message at the same trace site

        Adds any information to the message that poutines below it on the stack
        may need to execute properly.

        By default, does nothing, but overridden in derived classes.
        """
        return msg

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: a sample from the stochastic function at the site.

        Implements default pyro.sample Poutine behavior:
        if the observation at the site is not None, return the observation;
        else call the function and return the result.

        Derived classes often compute a side effect,
        then call super(Derived, self)._pyro_sample(msg).
        """
        fn, args, kwargs = \
            msg["fn"], msg["args"], msg["kwargs"]

        # msg["done"] enforces the guarantee in the poutine execution model
        # that a site's non-effectful primary computation should only be executed once:
        # if the site already has a stored return value,
        # don't reexecute the function at the site,
        # and do any side effects using the stored return value.
        if msg["done"]:
            return msg["value"]

        if msg["is_observed"]:
            assert msg["value"] is not None
            val = msg["value"]
        else:
            val = fn(*args, **kwargs)

        # after fn has been called, update msg to prevent it from being called again.
        msg["done"] = True
        return val

    def _pyro_param(self, msg):
        """
        :param msg: current message at a trace site.
        :returns: the result of querying the parameter store

        Implements default pyro.param Poutine behavior:
        queries the parameter store with the site name and varargs
        and returns the result of the query.

        If the parameter doesn't exist, create it using the site varargs.
        If it does exist, grab it from the parameter store.

        Derived classes often compute a side effect,
        then call super(Derived, self)._pyro_param(msg).
        """
        name, args, kwargs = \
            msg["name"], msg["args"], msg["kwargs"]

        # msg["done"] enforces the guarantee in the poutine execution model
        # that a site's non-effectful primary computation should only be executed once:
        # if the site already has a stored return value,
        # don't reexecute the function at the site,
        # and do any side effects using the stored return value.
        if msg["done"]:
            return msg["value"]

        ret = _PYRO_PARAM_STORE.get_param(name, *args, **kwargs)

        # after the param store has been queried, update msg["done"]
        # to prevent it from being queried again.
        msg["done"] = True
        return ret
