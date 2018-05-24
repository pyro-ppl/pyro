import functools

from .messenger import Messenger


# def stochastic(fn=None, name=None):
#     if fn is None:
#         return lambda f: stochastic(fn=f, name=name)
#
#     if name is None and fn is not None:
#         name = fn.__name__
#     scope_wrapped = poutine.scope(fn, prefix=name)
#
#     def _wrapped(*args, **kwargs):
#         return sample(name, scope_wrapped, *args, **kwargs)
#     return _wrapped


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
