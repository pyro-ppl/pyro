from __future__ import absolute_import, division, print_function

from greenlet import greenlet

from pyro.params import _PYRO_PARAM_STORE

# the global pyro stack
_PYRO_STACK = []


def am_i_wrapped():
    """
    :returns: True iff the currently executing code is wrapped in a poutine
    """
    return greenlet.getcurrent().parent is not None


def send_message(msg):
    """
    :param msg: a message to be sent
    :returns: reply message
    Sends a message to encapsulating poutine.
    """
    assert am_i_wrapped()
    return greenlet.getcurrent().parent.switch(msg)


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
            c = greenlet(self.fn)
            t = c.switch(*args, **kwargs)
            while not c.dead:
                msg = t
                self._process_message(msg)
                if am_i_wrapped() and not msg["stop"]:
                    reply = greenlet.getcurrent().parent.switch(msg)
                else:
                    reply = msg
                t = c.switch(reply)
        return t

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

    def _reset(self):
        """
        Resets the computation to the beginning, un-sampling all sample sites.

        By default, does nothing, but overridden in derived classes.
        """
        pass

    def _process_message(self, msg):
        """
        :param msg: current message at a trace site
        :returns: modified message after performing appropriate operations
        """
        if "reset" in msg and msg["reset"]:
            self._reset()
        else:
            return getattr(self, "_pyro_{}".format(msg["type"]))(msg)

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
            msg["value"] = val

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
        msg["value"] = ret

        # after the param store has been queried, update msg["done"]
        # to prevent it from being queried again.
        msg["done"] = True
        return ret
