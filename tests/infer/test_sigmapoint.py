from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from tests.common import assert_equal


@pytest.mark.parametrize("enumerate_", [None, "parallel"])
def test_dirichlet_inference(enumerate_):
    prior = torch.tensor([0.5, 1.0, 2.0])
    data = torch.tensor([3., 1., 2.])
    total_count = int(data.sum())
    true_posterior = prior + data

    def model(data):
        concentration = pyro.param("concentration", prior,
                                   constraint=constraints.positive)
        pyro.sample("prior", dist.Gamma(prior, 1.0).to_event(1), obs=concentration)
        probs = pyro.sample("probs", dist.SigmaPointDirichlet(concentration),
                            infer={"enumerate": enumerate_, "expand": False})
        pyro.sample("obs", dist.Multinomial(total_count, probs), obs=data)

    def guide(data):
        if not enumerate_:
            posterior = pyro.param("posterior", prior,
                                   constraint=constraints.positive)
            pyro.sample("probs", dist.Dirichlet(posterior))

    optim = Adam({"lr": 0.05})
    svi = SVI(model, guide, optim, TraceEnum_ELBO(strict_enumeration_warning=False))
    for step in range(101):
        loss = svi.step(data)
        if step % 10 == 0:
            print("step {} loss = {}".format(step, loss))

    actual = pyro.param("concentration")
    assert_equal(true_posterior, actual, prec=1e-2, msg="\n".join([
        "Expected: {}".format(true_posterior.detach().cpu().numpy()),
        "Actual: {}".format(actual.detach().cpu().numpy()),
    ]))
