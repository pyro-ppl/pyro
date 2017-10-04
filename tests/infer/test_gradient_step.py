import torch
import torch.optim
from torch.autograd import Variable
import pytest

import numpy as np
import pyro
import pyro.distributions as dist
from pyro.infer.kl_qp import KL_QP
from pyro.infer.tracegraph_kl_qp import TraceGraph_KL_QP
from pyro.util import ng_ones, ng_zeros


# The golden values below (mu_q_expected/log_sig_q_expected/) need to be updated each time
# KL_QP changes its random algorithm.
# If this leads to too much churn, simply delete this test.

@pytest.mark.init(rng_seed=0)
@pytest.mark.parametrize("kl_qp", [KL_QP, TraceGraph_KL_QP])
@pytest.mark.parametrize("reparameterized", [True, False])
def test_kl_qp_gradient_step_golden(kl_qp, reparameterized):
    verbose = True
    pyro.get_param_store().clear()
    mu_q_expected = {True: -1.1780080795288086, False: -1.178008079528809}[reparameterized]
    log_sig_q_expected = {True: -0.30474236607551575, False: -0.30474188923835754}[reparameterized]
    tolerance = 1.0e-7

    def model():
        mu_latent = pyro.sample("mu_latent", dist.diagnormal, ng_zeros(1), ng_ones(1))
        pyro.observe('obs', dist.diagnormal, Variable(torch.Tensor([0.23])), mu_latent, ng_ones(1))
        return mu_latent

    def guide():
        mu_q = pyro.param("mu_q", Variable(torch.randn(1), requires_grad=True))
        log_sig_q = pyro.param("log_sig_q", Variable(torch.randn(1), requires_grad=True))
        sig_q = torch.exp(log_sig_q)
        return pyro.sample("mu_latent", dist.diagnormal, mu_q, sig_q, reparameterized=reparameterized)

    kl_optim = kl_qp(model, guide, pyro.optim(torch.optim.Adam, {"lr": 0.1}))
    kl_optim.step()
    new_mu_q = pyro.param("mu_q").data.numpy()[0]
    new_log_sig_q = pyro.param("log_sig_q").data.numpy()[0]

    if verbose:
        print("after one step mu_q was %.15f; expected %.15f" % (new_mu_q, mu_q_expected))
        print("after one step log_sig_q was %.15f expected %.15f" % (new_log_sig_q, log_sig_q_expected))

    assert np.fabs(new_mu_q - mu_q_expected) < tolerance
    assert np.fabs(new_log_sig_q - log_sig_q_expected) < tolerance
