from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
from pyro.infer.discrete import infer_discrete
from tests.common import assert_close


def test_dirichlet_single():
    num_samples = 10000
    data = dist.Multinomial(10, torch.tensor([0.2, 0.4, 0.4])).sample()
    prior = torch.tensor([0.5, 1.0, 2.0])

    @infer_discrete
    def model(data):
        with pyro.plate("samples", num_samples):
            probs = pyro.sample("probs", dist.SigmaPointDirichlet(prior),
                                infer=dict(enumerate='parallel', expand=True))
            probs = probs / probs.sum(-1, True)
            pyro.sample("obs", dist.Multinomial(probs=probs), obs=data)
            return probs

    samples = model(data)
    posterior = dist.Dirichlet(prior + data)
    assert_close(samples.mean(0), posterior.mean, rtol=1e-2)
    assert_close(samples.var(0), posterior.variance, rtol=1e-2)
