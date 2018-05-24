import functools

from .poutine import Messenger
from .runtime import _PYRO_STACK


class ScopeMessenger(Messenger):
    def __init__(self, prefix=None):
        super(ScopeMessenger, self).__init__()
        self.prefix = prefix
        self._ref_count = 0

    def __enter__(self):
        if self not in _PYRO_STACK:
            super(ScopeMessenger, self).__enter__()
        self._ref_count += 1
        return self

    def __exit__(self, *args):
        self._ref_count -= 1
        if self._ref_count == 0:
            super(ScopeMessenger, self).__exit__()

    def __call__(self, fn):
        if self.prefix is None:
            self.prefix = fn.__name__

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            with self:
                return fn(*args, **kwargs)
        return _fn

    def _pyro_sample(self, msg):
        prefix = "/".join([self.prefix] * self._ref_count)
        msg["name"] = "{}/{}".format(prefix, msg["name"])
        return None
