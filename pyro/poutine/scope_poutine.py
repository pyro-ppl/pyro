import functools

from .poutine import Messenger, Poutine, _PYRO_STACK


def get_scope_stack():
    return list(map(lambda x: (x.prefix, x.suffix),
                    filter(lambda x: isinstance(x, ScopeMessenger),
                           _PYRO_STACK)))


class ScopeMessenger(Messenger):
    def __init__(self, prefix=None, suffix=None):
        super(ScopeMessenger, self).__init__()
        self.prefix = prefix
        self.suffix = suffix

    def __call__(self, fn):
        if self.prefix is None:
            self.prefix = fn.__name__

        # return functools.wraps(fn)(ScopePoutine(fn, prefix=self.prefix))
        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            with ScopeMessenger(prefix=self.prefix):
                return fn(*args, **kwargs)
        return _fn

    def _pyro_sample(self, msg):
        if self.suffix is not None:
            msg["name"] = "{}_{}/{}".format(
                self.prefix, self.suffix, msg["name"])
        else:
            msg["name"] = "{}/{}".format(self.prefix, msg["name"])
        return None  # return super(ScopeMessenger, self)._pyro_sample(msg)

    def _pyro_param(self, msg):
        if self.suffix is not None:
            msg["name"] = "{}_{}/{}".format(
                self.prefix, self.suffix, msg["name"])
        else:
            msg["name"] = "{}/{}".format(self.prefix, msg["name"])
        return None  # return super(ScopeMessenger, self)._pyro_param(msg)


class ScopePoutine(Poutine):
    def __init__(self, fn, *args, **kwargs):
        msngr = ScopeMessenger(*args, **kwargs)
        super(ScopePoutine, self).__init__(msngr, fn)
