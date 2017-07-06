# https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python
# unbounded memoize
# alternate in py3: https://docs.python.org/3/library/functools.html
# lru_cache
def memoize(fn):
    _mem = {}
    def _fn(*args, **kwargs):
        if (args, kwargs) not in _mem:
            _mem[(args, kwargs)] = fn(*args, **kwargs)
        return _mem[(args, kwargs)]
    return _fn


def ones(*args, **kwargs):
    return Parameter(torch.ones(*args, **kwargs))
    # return pyro.device(Parameter(torch.ones(*args, **kwargs)))


def zeros(*args, **kwargs):
    return Parameter(torch.zeros(*args, **kwargs))
    # return pyro.device(Parameter(torch.zeros(*args, **kwargs)))


def ng_ones(*args, **kwargs):
    return Variable(torch.ones(*args, **kwargs), requires_grad=False)


def ng_zeros(*args, **kwargs):
    return Variable(torch.zeros(*args, **kwargs), requires_grad=False)


