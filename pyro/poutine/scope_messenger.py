import functools

from .messenger import Messenger


class ScopeMessenger(Messenger):
    def __init__(self, prefix=None):
        super(ScopeMessenger, self).__init__()
        self.prefix = prefix

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
