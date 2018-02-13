import functools

from .poutine import Messenger, Poutine


class ScopeMessenger(Messenger):
    def __init__(self, prefix=None, multi=None):
        super(ScopeMessenger, self).__init__()
        if multi is None:
            multi = False
        self.prefix = prefix
        self.multi = multi
        self.n_calls = 1  # {}

    def __call__(self, fn):
        if self.prefix is None:
            self.prefix = fn.__name__

        @functools.wraps(fn)
        def _fn(*args, **kwargs):
            # keep n_calls dict for each context
            # each time called in same context, increment n_calls[ctx]
            # reset all n_calls entries after returning to global context
            # extra help: each context can only happen once per global context
            with ScopeMessenger(prefix=self.prefix, multi=self.multi) as cm:
                if self.multi:
                    cm.n_calls = int(self.n_calls)
                    self.n_calls += 1
                return fn(*args, **kwargs)
        return _fn

    def __exit__(self, *args, **kwargs):
        self.n_calls += 1
        return super(ScopeMessenger, self).__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        # check the context in the stack
        # if ctx in n_calls, update n_calls[ctx] += 1
        # if ctx not in n_calls, set n_calls[ctx] = 0
        if self.multi:
            msg["name"] = "{}_{}/{}".format(
                self.prefix, self.n_calls, msg["name"])
        else:
            msg["name"] = "{}/{}".format(self.prefix, msg["name"])
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
