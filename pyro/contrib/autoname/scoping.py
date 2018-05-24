"""
``pyro.contrib.autoname.scoping`` contains the implementation of
:func:`pyro.contrib.autoname.scope`, a tool for automatically appending
a semantically meaningful prefix to names of sample sites.
"""
import functools

from pyro.poutine.messenger import Messenger


class ScopeMessenger(Messenger):
    def __init__(self, prefix=None):
        super(ScopeMessenger, self).__init__()
        self.prefix = prefix

    def __enter__(self):
        if self.prefix is None:
            raise ValueError("no prefix was provided")
        return super(ScopeMessenger, self).__enter__()

    def __call__(self, fn):
        if self.prefix is None:
            self.prefix = fn.__name__

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            with ScopeMessenger(prefix=self.prefix):
                return fn(*args, **kwargs)
        return _fn

    def _pyro_sample(self, msg):
        msg["name"] = "{}/{}".format(self.prefix, msg["name"])
        return None


def scope(fn=None, prefix=None):
    """
    scope
    """
    msngr = ScopeMessenger(prefix=prefix)
    return msngr(fn) if fn is not None else msngr


__all__ = [
    "scope",
]
