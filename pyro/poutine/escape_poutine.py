from __future__ import absolute_import, division, print_function

from pyro.util import NonlocalExit

from .poutine import Poutine


class EscapePoutine(Poutine):
    """
    Poutine that does a nonlocal exit by raising a util.NonlocalExit exception
    """
    def __init__(self, fn, escape_fn):
        """
        :param fn: a stochastic function (callable containing pyro primitive calls)
        :param escape_fn: function that takes a msg as input and returns True
            if the poutine should perform a nonlocal exit at that site.

        Constructor.  Stores fn and escape_fn.
        """
        self.escape_fn = escape_fn
        super(EscapePoutine, self).__init__(fn)

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
            raise NonlocalExit(msg)
        else:
            return super(EscapePoutine, self)._pyro_sample(msg)
