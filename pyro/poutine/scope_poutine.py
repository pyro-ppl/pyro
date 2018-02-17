import functools

from .poutine import Messenger, Poutine


class ScopeMessenger(Messenger):
    def __init__(self, prefix=None, suffix=None):
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

    def __exit__(self, *args, **kwargs):
        self.n_calls += 1
        return super(ScopeMessenger, self).__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        msg["name"] = "{}_{}/{}".format(self.prefix, self.suffix, msg["name"])
        return None  # return super(ScopeMessenger, self)._pyro_sample(msg)

    def _pyro_param(self, msg):
        if self.multi:
            msg["name"] = "{}_{}/{}".format(
                self.prefix, self.n_calls, msg["name"])
        else:
            msg["name"] = "{}/{}".format(self.prefix, msg["name"])
        return None  # return super(ScopeMessenger, self)._pyro_param(msg)


class ScopePoutine(Poutine):
    def __init__(self, fn, *args, **kwargs):
        msngr = ScopeMessenger(*args, **kwargs)
        super(ScopePoutine, self).__init__(msngr, fn)
