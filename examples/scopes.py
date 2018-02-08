import pyro
import pyro.util
import pyro.poutine as poutine
import pyro.distributions as dist

import contextlib
import functools
import torch
from torch.autograd import Variable


def scope(prefix=None, multi=None, flat=None):
    """
    XXX
    """
    def scope_decorator(fn):
        _fn = poutine.ScopePoutine(fn, prefix=prefix, multi=multi)

        @functools.wraps(fn)
        def scope_wrapper(*args, **kwargs):
            # return ScopePoutine(fn, prefix=prefix, multi=multi, flat=flat)(*args, **kwargs)
            return _fn(*args, **kwargs)
        return scope_wrapper

    return scope_decorator


def scope2(fn, prefix=None, multi=None, flat=None):
    return poutine.ScopePoutine(fn, prefix=prefix, multi=multi, flat=flat)


@scope()  # prefix="latent1")
def model1():
    model2()
    model2()


@scope(multi=True)  # , prefix="latent2")
def model2():
    return pyro.sample("y", dist.normal,
                       Variable(torch.zeros(1)),
                       Variable(torch.ones(1)))


@scope()
def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x{}".format(t), dist.bernoulli, p)
    if x.data[0] == 1.0:
        return x + geometric(p, t=t+1)
    else:
        return x


# print(poutine.trace(model1).get_trace().nodes)
print(poutine.trace(geometric).get_trace(Variable(torch.Tensor([0.7]))).nodes)
