from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch
import torch.nn as nn

import dmm.polyphonic_data_loader as poly
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings
from torch.distributions import constraints


# sizes etc.
N_c = 97  # number of territories
N_f = 5   # number of features (?)
N_v = 10  # number of random effect values (K in the paper)
s_per_t = torch.ones(97) * 8
N_s = 8
N_t = 11


# @config_enumerate
def model(x, z1, z2, theta):
    # transition matrix GLM parameters
    beta = pyro.param("beta", torch.zeros((2, N_f)))

    # random effect probs
    probs_e = pyro.param("probs_e", torch.ones((N_v,)), constraint=constraints.simplex)

    # likelihood parameters
    # # turn angle
    # p_z1 = pyro.param("p_z1", ...)
    # conc_z1 = pyro.param("conc_z1", ...)
    # step length
    a_z2 = pyro.param("a_z2", torch.tensor([1., 1.]), constraint=constraints.positive)
    b_z2 = pyro.param("b_z2", torch.tensor([1., 1.]), constraint=constraints.positive)

    with pyro.plate("territory", N_c) as c:
        e = pyro.sample("e", dist.Categorical(probs_e))
        eps = theta[e]
        with pyro.plate("session", N_s) as s:  # s_per_t.max()) as s:  # , poutine.mask(s > s_per_t[c]):
            gamma = eps.t() + beta.mm(x)
            y = 0
            for t in pyro.markov(range(N_t)):  # max(times))):  # TODO get max right
                y = pyro.sample("y_{}".format(t), dist.Bernoulli(logits=gamma[y])).long()
                # bivariate observations
                # pyro.sample("z1_{}".format(t), dist.VonMises(p_z1[y]), obs=z1[t])
                # TODO replace Gamma with zero-inflated Gamma or masked mixture
                pyro.sample("z2_{}".format(t), dist.Gamma(a_z2[y], b_z2[y]), obs=z2[t])
    return z1, z2


def test_model():
    # random effect values - fixed for now
    theta = torch.randn((N_v, 2))
    x = torch.randn((N_f, N_c))
    z1, z2 = model(x, [None] * N_t, [None] * N_t, theta)


if __name__ == "__main__":
    test_model()
