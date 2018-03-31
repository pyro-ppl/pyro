from __future__ import absolute_import, division, print_function

from .poutine import Messenger, Poutine


class ContinuationMessenger(Messenger):
    """
    This Messenger sets the continuation field at each site,
    and makes sure it gets called and unwound correctly.
    Continuations give up (or simulate giving up) control of execution
    by expanding along a dimension, yielding a coroutine,
    or raising a special poutine.util.NonlocalExit exception.

    It is currently used primarily for sequential and parallel enumeration.

    Generalizes and replaces EscapeMessenger and EnumerateMessenger.
    """
    def __init__(self, escape_fn, cont_fn, first_available_dim):
        """
        :param escape_fn: boolean function evaluated on site
        :param cont_fn: function of site that returns None or raises NonlocalExit
        :param first_available_dim: first available expansion dimension

        Sets escape_fn, the function evaluated at each site to decide
        whether to apply cont_fn, the continuation.
        If cont_fn is applied, mark available dimensions starting from
        first_available_dim.
        """
        if first_available_dim is None:
            first_available_dim = float('inf')
        self.escape_fn = escape_fn
        self.cont_fn = cont_fn
        self.first_available_dim = first_available_dim
        self.next_available_dim = None

    def __enter__(self):
        """
        Resets the next available expansion dimension.
        """
        self.next_available_dim = self.first_available_dim
        return super(ContinuationMessenger, self).__enter__()

    def _postprocess_message(self, msg):
        """
        Increments the next available expansion dimension.
        """
        if "next_available_dim" in msg["infer"]:
            self.next_available_dim = msg["infer"]["next_available_dim"]

    def _pyro_sample(self, msg):
        """
        If self.escape_fn evaluates to True
        and a continuation hasn't already been applied at this site,
        set the continuation field of the site and mark it done.
        """
        if self.escape_fn(msg) and not msg["done"]:
            msg["infer"]["next_available_dim"] = self.next_available_dim
            msg["done"] = True
            msg["continuation"] = self.cont_fn

    def _pyro_param(self, msg):
        """
        If self.escape_fn evaluates to True
        and a continuation hasn't already been applied at this site,
        set the continuation field of the site and mark it done.
        """
        if self.escape_fn(msg) and not msg["done"]:
            msg["done"] = True
            msg["continuation"] = self.cont_fn


class ContinuationPoutine(Poutine):
    """
    This Poutine sets the continuation field at each site,
    and makes sure it gets called and unwound correctly.
    Continuations give up (or simulate giving up) control of execution
    by expanding along a dimension, yielding a coroutine,
    or raising a special poutine.util.NonlocalExit exception.

    It is currently used primarily for sequential and parallel enumeration.

    Generalizes and replaces EscapePoutine and EnumeratePoutine.
    """
    def __init__(self, fn, escape_fn, cont_fn, first_available_dim=None):
        """
        :param escape_fn: boolean function evaluated on site
        :param cont_fn: function of site that returns None or raises NonlocalExit
        :param first_available_dim: first available expansion dimension

        Sets escape_fn, the function evaluated at each site to decide
        whether to apply cont_fn, the continuation.
        If cont_fn is applied, mark available dimensions starting from
        first_available_dim.
        """
        super(ContinuationPoutine, self).__init__(
            ContinuationMessenger(escape_fn, cont_fn, first_available_dim), fn)
