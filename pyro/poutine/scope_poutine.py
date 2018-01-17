from .poutine import Poutine


class ScopePoutine(Poutine):
    def __init__(self, fn, prefix=None, multi=None, flat=None, **kwargs):
        if multi is None:
            multi = False
        if flat is None:
            flat = False
        if prefix is None:
            prefix = fn.__name__
        self.prefix = prefix
        self.multi = multi
        self.n_calls = 0
        super(ScopePoutine, self).__init__(fn, **kwargs)

    def __exit__(self, *args, **kwargs):
        self.n_calls += 1
        return super(ScopePoutine, self).__exit__(*args, **kwargs)

    def _pyro_sample(self, msg):
        if self.multi:
            msg["name"] = "{}_{}/{}".format(
                self.prefix, self.n_calls, msg["name"])
        else:
            msg["name"] = "{}/{}".format(self.prefix, msg["name"])
        return super(ScopePoutine, self)._pyro_sample(msg)

    def _pyro_param(self, msg):
        if self.multi:
            msg["name"] = "{}_{}/{}".format(
                self.prefix, self.n_calls, msg["name"])
        else:
            msg["name"] = "{}/{}".format(self.prefix, msg["name"])
        return super(ScopePoutine, self)._pyro_param(msg)
