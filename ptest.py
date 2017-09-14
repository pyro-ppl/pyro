from __future__ import print_function
import numpy as np
import torch
import pdb
import sys
from torch.autograd import Variable
if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue

import pyro
import pyro.distributions as d
from pyro.distributions import DiagNormal, Bernoulli
import pyro.poutine as poutine
from pyro.util import memoize

# print(poutine.__dict__.keys())
def model():
    latent1 = pyro.sample("latent1",
                          DiagNormal(Variable(torch.zeros(2)),
                                     Variable(torch.ones(2))))
    latent2 = pyro.sample("latent2",
                          DiagNormal(latent1,
                                     5 * Variable(torch.ones(2))))
    x_dist = DiagNormal(latent2, Variable(torch.ones(2)))
    x = pyro.observe("obs", x_dist, Variable(torch.ones(2)))
    return latent1

def guide():
    mu1 = pyro.param("mu1", Variable(torch.randn(2), requires_grad=True))
    sigma1 = pyro.param("sigma1", Variable(torch.ones(2), requires_grad=True))
    latent1 = pyro.sample("latent1", DiagNormal(mu1, sigma1))
#    latent1 = pyro.sample("latent1", d.diagnormal, mu1, sigma1)
#     return latent1
#     mu2 = pyro.param("mu2", Variable(torch.randn(2), requires_grad=True))
#     sigma2 = pyro.param("sigma2", Variable(torch.ones(2), requires_grad=True))
#     latent2 = pyro.sample("latent2", DiagNormal(mu2, sigma2))
#     return latent2
guide_sites = ["latent1", "latent2",
                            "mu1", "sigma1",
                            "mu2", "sigma2",
                            "_INPUT", "_RETURN"]
print(poutine.trace(guide)())
# guide_trace = poutine.trace(guide)()
# queue = Queue()
# queue.put(poutine.Trace())
guide_trace = poutine.trace(poutine.block(guide,expose=guide_sites))()
print("KEYS",guide_trace.keys())
print('-----------------------')
# print(poutine.trace(poutine.queue(guide, queue=queue))())
print(poutine.trace(poutine.replay(model, guide_trace)))()
# print(poutine.trace(model)())
