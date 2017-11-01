from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import torch
import torch.optim
from torch.autograd import Variable

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI
from pyro.util import ng_ones, ng_zeros

# The golden values below (mu_q_expected/log_sig_q_expected/) need to be updated each time
# ELBO changes its random algorithm.
# If this leads to too much churn, simply delete this test.


@pytest.mark.init(rng_seed=0)
@pytest.mark.parametrize("trace_graph", [False, True])
@pytest.mark.parametrize("reparameterized", [True, False])
def test_kl_qp_gradient_step_golden(trace_graph, reparameterized):
    verbose = True
    pyro.clear_param_store()
    mu_q_expected = {True: -1.1780080795288086, False: -1.178008079528809}[reparameterized]
    log_sig_q_expected = {True: -0.30474236607551575, False: -0.30474188923835754}[reparameterized]
    tolerance = 1.0e-7

    def model():
        mu_latent = pyro.sample("mu_latent", dist.Normal(ng_zeros(1), ng_ones(1), reparameterized=reparameterized))
        pyro.observe('obs', dist.normal, Variable(torch.Tensor([0.23])), mu_latent, ng_ones(1))
        return mu_latent

    def guide():
        mu_q = pyro.param("mu_q", Variable(torch.randn(1), requires_grad=True))
        log_sig_q = pyro.param("log_sig_q", Variable(torch.randn(1), requires_grad=True))
        sig_q = torch.exp(log_sig_q)
        return pyro.sample("mu_latent", dist.Normal(mu_q, sig_q, reparameterized=reparameterized))

    adam = optim.Adam({"lr": .10})
    svi = SVI(model, guide, adam, loss="ELBO", trace_graph=trace_graph)
    svi.step()

    new_mu_q = pyro.param("mu_q").data.cpu().numpy()[0]
    new_log_sig_q = pyro.param("log_sig_q").data.cpu().numpy()[0]

    if verbose:
        print("\nafter one step mu_q was %.15f; expected %.15f" % (new_mu_q, mu_q_expected))
        print("after one step log_sig_q was %.15f expected %.15f" % (new_log_sig_q, log_sig_q_expected))

    if pyro.param("mu_q").is_cuda:
        # Ignore this case since cuda is too nondeterministic.
        pass
    else:
        assert np.fabs(new_mu_q - mu_q_expected) < tolerance
        assert np.fabs(new_log_sig_q - log_sig_q_expected) < tolerance
