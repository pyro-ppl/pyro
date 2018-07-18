from __future__ import absolute_import, division, print_function

from .messenger import Messenger
from .runtime import NonlocalExit


class EscapeMessenger(Messenger):
    """
    Given a callable that contains Pyro primitive calls,
    evaluate escape_fn on each site, and if the result is True,
    raise a :class:`~pyro.poutine.runtime.NonlocalExit` exception that stops execution
    and returns the offending site.

    :param fn: a stochastic function (callable containing Pyro primitive calls)
    :param escape_fn: function that takes a partial trace and a site,
        and returns a boolean value to decide whether to exit at that site
    :returns: stochastic function decorated with :class:`~pyro.poutine.escape_messenger.EscapeMessenger`
    """
    def __init__(self, escape_fn):
        """
        :param escape_fn: function that takes a msg as input and returns True
            if the poutine should perform a nonlocal exit at that site.

        Constructor.  Stores fn and escape_fn.
        """
        super(EscapeMessenger, self).__init__()
        self.escape_fn = escape_fn

    def _pyro_sample(self, msg):
        """
        :param msg: current message at a trace site
        :returns: a sample from the stochastic function at the site.

        Evaluates self.escape_fn on the site (self.escape_fn(msg)).

        If this returns True, raises an exception NonlocalExit(msg).
        Else, implements default _pyro_sample behavior with no additional effects.
        """
        if self.escape_fn(msg):
            msg["done"] = True
            msg["stop"] = True

            def cont(m):
                raise NonlocalExit(m)
            msg["continuation"] = cont
        return None
