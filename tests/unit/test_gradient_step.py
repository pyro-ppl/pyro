import torch
import torch.optim
from torch.autograd import Variable
import pytest

import pyro
import pyro.distributions as dist
from pyro.infer.kl_qp import KL_QP
from pyro.infer.tracegraph_kl_qp import TraceGraph_KL_QP
from pyro.util import ng_ones, ng_zeros

pytestmark = pytest.mark.init(rng_seed=0)


@pytest.mark.parametrize("kl_qp", [KL_QP, TraceGraph_KL_QP])
@pytest.mark.parametrize("reparameterized", [True, False])
def test_kl_qp_gradient_step(kl_qp, reparameterized):
    if kl_qp == KL_QP and not reparameterized:
        return
    verbose = True
    pyro.get_param_store().clear()
    mu_q_expected = {True: -1.2770081289993769, False: -1.2770081290294442}[reparameterized]
    log_sig_q_expected = {True: -0.40374189711367497, False: -0.40374189244231484}[reparameterized]

    def model():
        mu_latent = pyro.sample("mu_latent", dist.diagnormal, ng_zeros(1), ng_ones(1))
        pyro.observe('obs', dist.diagnormal, Variable(torch.Tensor([0.23])), mu_latent, ng_ones(1))
        return mu_latent

    def guide():
        mu_q = pyro.param("mu_q", Variable(torch.randn(1), requires_grad=True))
        log_sig_q = pyro.param("log_sig_q", Variable(torch.randn(1), requires_grad=True))
        sig_q = torch.exp(log_sig_q)
        return pyro.sample("mu_latent", dist.diagnormal, mu_q, sig_q, reparameterized=reparameterized)

    kl_optim = kl_qp(model, guide, pyro.optim(torch.optim.Adam, {"lr": 0.001}))
    kl_optim.step()
    if verbose:
        print("after one step mu_q was %.12f; expected %.12f" % (pyro.param("mu_q").data.numpy()[0],
                                                                 mu_q_expected))
        print("after one step log_sig_q was %.12f; expected %.12f" % (pyro.param("log_sig_q").data.numpy()[0],
                                                                      log_sig_q_expected))

    assert(pyro.param("mu_q").data.numpy()[0] == mu_q_expected)
    assert(pyro.param("log_sig_q").data.numpy()[0] == log_sig_q_expected)
