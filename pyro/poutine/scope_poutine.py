import functools

from .poutine import Messenger, Poutine, _PYRO_STACK


def get_scope_stack():
    return tuple(map(lambda x: (x.prefix, x.suffix),
                     filter(lambda x: isinstance(x, ScopeMessenger),
                            _PYRO_STACK)))


class _ScopeAllocator(object):
    """
    TODO docs
    """
    def __init__(self):
        self._stacks = {}

    def allocate(self, name, suffix, context):
        if context not in self._stacks:
            self._stacks[context] = {}
        if name not in self._stacks[context]:
            self._stacks[context][name] = set()
        self._stacks[context][name].add(suffix)
        return suffix + 1

    def free(self, name, suffix, context):
        # XXX remove all contexts where name is the root??
        self._stacks[context][name].discard(suffix)


_SCOPE_ALLOCATOR = _ScopeAllocator()


class ScopeMessenger(Messenger):
    def __init__(self, prefix=None, suffix=None):
        super(ScopeMessenger, self).__init__()
        self.prefix = prefix
        self.suffix = suffix

    def __enter__(self):
        self.context = get_scope_stack()
        self.suffix = _SCOPE_ALLOCATOR.allocate(self.name, self.suffix, self.context)
        return super(ScopeMessenger, self).__enter__()

    def __exit__(self, *args):
        _SCOPE_ALLOCATOR.free(self.name, self.suffix, self.context)
        return super(ScopeMessenger, self).__exit__(*args)

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
