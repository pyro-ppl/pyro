from pdb import set_trace as bb
import numpy as np
import torch
from torch.autograd import Variable

import pyro
from pyro.distributions import DiagNormal

from pyro.infer.abstract_infer import LikelihoodWeighting, lw_expectation
from pyro.infer.importance_sampling import ImportanceSampling
from pyro.infer.kl_qp import KL_QP


def model():
    latent = pyro.sample("latent",
                         DiagNormal(Variable(torch.zeros(1)),
                                    5 * Variable(torch.ones(1))))
    x_dist = DiagNormal(latent, Variable(torch.ones(1)))
    x = pyro.observe("obs", x_dist, Variable(torch.ones(1)))
    return latent


# now let's try inference
infer = LikelihoodWeighting(model)

exp = lw_expectation(infer, lambda x: x, 100)
print(exp)

# and try importance!


def guide():
    latent = pyro.sample("latent",
                         DiagNormal(Variable(torch.zeros(1)),
                                    5 * Variable(torch.ones(1))))
    x_dist = DiagNormal(latent, Variable(torch.ones(1)))
    pass


infer = ImportanceSampling(model, guide)

exp = lw_expectation(infer, lambda x: x, 100)
print(exp)


# and try VI!
def guide():
    mf_m = pyro.param("mf_m", Variable(torch.zeros(1)))
    mf_v = pyro.param("mf_v", Variable(torch.ones(1)))
    latent = pyro.sample("latent",
                         DiagNormal(mf_m, mf_v))
    pass


infer = KL_QP(model, guide)

exp = lw_expectation(infer, lambda x: x, 100)
print(exp)
