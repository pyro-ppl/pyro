import pyro
import pyro.util
import pyro.poutine as poutine
import pyro.distributions as dist

import torch
from torch.autograd import Variable

from pyro.poutine.scope_poutine import ScopeMessenger as scope


@scope()
def model1():
    model2()
    model2()


@scope(multi=True)
def model2():
    return pyro.sample("y", dist.normal,
                       Variable(torch.zeros(1)),
                       Variable(torch.ones(1)))


@scope()
def geometric(p):
    x = pyro.sample("x", dist.bernoulli, p)
    if x.data[0] == 1.0:
        return x + geometric(p)
    else:
        return x


print(poutine.trace(model2).get_trace().nodes)

print(poutine.trace(model1).get_trace().nodes)

print(poutine.trace(geometric).get_trace(Variable(torch.Tensor([0.7]))).nodes)
