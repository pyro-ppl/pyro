import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Parameter


def detach_iterable(iterable):
    if isinstance(iterable, Variable):
        return iterable.detach()
    else:
        return [var.detach() for var in iterable]


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


def log_beta(t):
    if t.dim() == 1:
        numer = torch.sum(log_gamma(t))
        denom = log_gamma(torch.sum(t))
    else:
        numer = torch.sum(log_gamma(t), 1)
        denom = log_gamma(torch.sum(t, 1))
    return numer - denom


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


def basic_histogram(ps, vs):
    """
    make a histogram from weighted things that aren't tensors
    Horribly slow...
    """
    assert isinstance(vs, (list, tuple)), \
        "vs must be a primitive type that preserves ordering at construction"
    hist = {}
    for i, v in enumerate(vs):
        if v not in hist:
            hist[v] = 0.0
        hist[v] = hist[v] + ps[i]
    return {"ps": torch.cat([hist[v] for v in hist.keys()]),
            "vs": [v for v in hist.keys()]}


def get_batch_indices(data, batch_size, batch_dim):
    """
    Compute batch indices used for subsampling in map_data
    Weirdly complicated because of type ambiguity
    """
    if isinstance(data, (torch.Tensor, Variable)):  # XXX and np.ndarray?
        assert batch_dim >= 0, \
            "batch_dim must be nonnegative"
        assert batch_size <= data.size(batch_dim), \
            "batch must be smaller than dataset size"
        if batch_size > 0:
            ind = Variable(torch.randperm(data.size(batch_dim))[0:batch_size])
        else:
            # if batch_size == 0, don't index (saves time/space)
            ind = Variable(torch.arange(0, data.size(batch_dim)))
    else:
        # handle lists and other ordered sequence types (e.g. tuples but not sets)
        assert batch_dim == 0, \
            "batch dim for non-tensor map_data must be 0"
        assert batch_size <= len(data), \
            "batch must be smaller than dataset size"
        # if batch_size > 0, select a random set of indices and store it
        if batch_size > 0:
            ind = torch.randperm(len(data))[0:batch_size].numpy().tolist()
        else:
            ind = list(range(len(data)))

    return ind


def get_batch_scale(data, batch_size, batch_dim):
    """
    Compute scale used for subsampling in map_data
    Weirdly complicated because of type ambiguity
    """
    if isinstance(data, (torch.Tensor, Variable)):  # XXX and np.ndarray?
        assert batch_size <= data.size(batch_dim), \
            "batch must be smaller than dataset size"
        if batch_size > 0:
            scale = float(data.size(batch_dim)) / float(batch_size)
        else:
            # if batch_size == 0, don't index (saves time/space)
            scale = 1.0
    else:
        assert batch_size <= len(data), \
            "batch must be smaller than dataset size"
        # if batch_size > 0, select a random set of indices and store it
        if batch_size > 0:
            scale = float(len(data)) / float(batch_size)
        else:
            scale = 1.0

    return scale
