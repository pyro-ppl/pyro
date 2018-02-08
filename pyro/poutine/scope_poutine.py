from .poutine import Messenger, Poutine


class ScopeMessenger(Messenger):
    def __init__(self, prefix=None, multi=None, flat=None):
        super(ScopeMessenger, self).__init__()
        if multi is None:
            multi = False
        if flat is None:
            flat = False
        # if prefix is None:
        #     prefix = fn.__name__
        self.prefix = prefix
        self.multi = multi
        self.n_calls = 0

    def __exit__(self, *args, **kwargs):
        self.n_calls += 1
        return super(ScopeMessenger, self).__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        if self.multi:
            msg["name"] = "{}_{}/{}".format(
                self.prefix, self.n_calls, msg["name"])
        else:
            msg["name"] = "{}/{}".format(self.prefix, msg["name"])
        return super(ScopeMessenger, self)._pyro_sample(msg)

    def _pyro_param(self, msg):
        if self.multi:
            msg["name"] = "{}_{}/{}".format(
                self.prefix, self.n_calls, msg["name"])
        else:
            msg["name"] = "{}/{}".format(self.prefix, msg["name"])
        return super(ScopeMessenger, self)._pyro_param(msg)


class ScopePoutine(Poutine):
    def __init__(self, fn, *args, **kwargs):
        msngr = ScopeMessenger(*args, **kwargs)
        super(ScopePoutine, self).__init__(msngr, fn)
