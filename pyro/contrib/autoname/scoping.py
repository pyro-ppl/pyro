"""
``pyro.contrib.autoname.scoping`` contains the implementation of
:func:`pyro.contrib.autoname.scope`, a tool for automatically appending
a semantically meaningful prefix to names of sample sites.
"""
import functools

from pyro.poutine.messenger import Messenger
from pyro.poutine.runtime import apply_stack


class ScopeMessenger(Messenger):
    """
    TODO docs
    """
    def __init__(self, prefix=None, inner=None):
        super(ScopeMessenger, self).__init__()
        self.prefix = prefix
        self.inner = inner

    def __enter__(self):
        if self.prefix is None:
            raise ValueError("no prefix was provided")
        if not self.inner:
            msg = {
                "type": "sample",
                "name": self.prefix,
                "fn": lambda: 0.,
                "is_observed": False,
                "args": (),
                "kwargs": {},
                "value": None,
                "infer": {},
                "scale": 1.0,
                "cond_indep_stack": (),
                "done": False,
                "stop": False,
                "continuation": None,
                "PRUNE": True
            }
            apply_stack(msg)
            self.prefix = msg["name"].split("/")[-1]
        return super(ScopeMessenger, self).__enter__()

    def __call__(self, fn):
        if self.prefix is None:
            self.prefix = fn.__code__.co_name  # fn.__name__

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            with type(self)(prefix=self.prefix, inner=self.inner):
                return fn(*args, **kwargs)
        return _fn

    def _pyro_sample(self, msg):
        msg["name"] = "{}/{}".format(self.prefix, msg["name"])
        return None


def scope(fn=None, prefix=None, inner=None):
    """
    scope
    """
    msngr = ScopeMessenger(prefix=prefix, inner=inner)
    return msngr(fn) if fn is not None else msngr
