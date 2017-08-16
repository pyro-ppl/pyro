import numpy as np
import copy
import torch
from torch.autograd import Variable
from torch.nn import Parameter


def _dict_to_tuple(d):
    """
    Recursively converts a dictionary to a list of key-value tuples
    Only intended for use as a helper function inside memoize!!
    May break when keys cant be sorted, but that is not an expected use-case
    """
    if isinstance(d, dict):
        return tuple([(k, _dict_to_tuple(d[k])) for k in sorted(d.keys())])
    else:
        return d


def memoize(fn):
    """
    https://stackoverflow.com/questions/1988804/what-is-memoization-and-how-can-i-use-it-in-python
    unbounded memoize
    alternate in py3: https://docs.python.org/3/library/functools.html
    lru_cache
    """
    _mem = {}

    def _fn(*args, **kwargs):
        kwargs_tuple = _dict_to_tuple(kwargs)
        if (args, kwargs_tuple) not in _mem:
            _mem[(args, kwargs_tuple)] = fn(*args, **kwargs)
        return _mem[(args, kwargs_tuple)]
    return _fn


def ones(*args, **kwargs):
    return Parameter(torch.ones(*args, **kwargs))


def zeros(*args, **kwargs):
    return Parameter(torch.zeros(*args, **kwargs))


def ng_ones(*args, **kwargs):
    return Variable(torch.ones(*args, **kwargs), requires_grad=False)


def ng_zeros(*args, **kwargs):
    return Variable(torch.zeros(*args, **kwargs), requires_grad=False)


def log_sum_exp(vecs):
    n = len(vecs.size())
    if n == 1:
        vecs = vecs.view(1, -1)
    _, idx = torch.max(vecs, 1)
    max_score = torch.index_select(vecs, 1, idx.view(-1))
    ret = max_score + torch.log(torch.sum(torch.exp(vecs - max_score.expand_as(vecs))))
    if n == 1:
        return ret.view(-1)
    return ret


def zero_grads(tensors):
    """
    Sets gradients of list of Variables to zero in place
    """
    for p in tensors:
        if p.grad is not None:
            if p.grad.volatile:
                p.grad.data.zero_()
            else:
                data = p.grad.data
                p.grad = Variable(data.new().resize_as_(data).zero_())


def log_gamma(xx):
    if isinstance(xx, Variable):
        ttype = xx.data.type()
    elif isinstance(xx, torch.Tensor):
        ttype = xx.type()
    gamma_coeff = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5
    ]
    magic1 = 1.000000000190015
    magic2 = 2.5066282746310005
    x = xx - 1.0
    t = x + 5.5
    t = t - (x + 0.5) * torch.log(t)
    ser = Variable(torch.ones(x.size()).type(ttype)) * magic1
    for c in gamma_coeff:
        x = x + 1.0
        ser = ser + torch.pow(x / c, -1)
    return torch.log(ser * magic2) - t


def to_one_hot(x, ps):
    if isinstance(x, Variable):
        ttype = x.data.type()
    elif isinstance(x, torch.Tensor):
        ttype = x.type()
    batch_size = x.size(0)
    classes = ps.size(1)
    # create an empty array for one-hots
    batch_one_hot = torch.zeros(batch_size, classes)
    # this operation writes ones where needed
    batch_one_hot.scatter_(1, x.data.view(-1, 1).long(), 1)

    return Variable(batch_one_hot.type(ttype))


def tensor_histogram(ps, vs):
    """
    make a histogram from weighted Variable/Tensor/ndarray samples
    Horribly slow...
    """
    # first, get everything into the same form: numpy arrays
    np_vs = []
    for v in vs:
        _v = v
        if isinstance(_v, Variable):
            _v = _v.data
        if isinstance(_v, torch.Tensor):
            _v = _v.numpy()
        np_vs.append(_v)
    # now form the histogram
    hist = dict()
    for p, v, np_v in zip(ps, vs, np_vs):
        k = tuple(np_v.flatten().tolist())
        if k not in hist:
            # XXX should clone?
            hist[k] = [0.0, v]
        hist[k][0] = hist[k][0] + p
    # now split into keys and original values
    ps2 = []
    vs2 = []
    for k in hist.keys():
        ps2.append(hist[k][0])
        vs2.append(hist[k][1])
    # return dict suitable for passing into Categorical
    return {"ps": torch.cat(ps2), "vs": np.array(vs2).flatten()}


def rescale_dist(fn, scale):
    """
    Rescales a distribution's log_pdf method output by a constant factor
    """
    if hasattr(fn, "log_pdf"):
        old_log_pdf = fn.log_pdf

        def new_log_pdf(*args, **kwargs):
            return old_log_pdf(*args, **kwargs) * scale

        new_fn = copy.copy(fn)  # XXX incorrect?
        new_fn.log_pdf = new_log_pdf
        return new_fn
    else:
        # XXX should raise an error here?
        return fn
