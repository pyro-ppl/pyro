from __future__ import absolute_import, division, print_function

import pytest

import pyro
from pyro.infer.tracegraph_elbo import TraceGraph_ELBO
from tests.common import assert_equal
from tests.integration_tests.test_conjugate_gaussian_models import GaussianChain


# TODO increase precision and number of particles once latter is parallelized properly
@pytest.mark.parametrize("N", [3, 5])
@pytest.mark.parametrize("reparameterized", [True, False])
def test_conjugate_chain_gradient(N, reparameterized):
    pyro.clear_param_store()

    gc = GaussianChain()
    gc.setUp()
    gc.setup_chain(N)

    elbo = TraceGraph_ELBO(num_particles=1000)
    elbo.loss_and_grads(gc.model, gc.guide, reparameterized=reparameterized)

    for i in range(1, N + 1):
        for param_prefix in ["mu_q_%d", "log_sig_q_%d", "kappa_q_%d"]:
            if i == N and param_prefix == 'kappa_q_%d':
                continue
            actual_grad = pyro.param(param_prefix % i).grad
            assert_equal(actual_grad, 0.0 * actual_grad, prec=0.20, msg="".join([
                         "parameter %s%d" % (param_prefix[:-2], i),
                         "\nexpected = zero vector",
                         "\n  actual = {}".format(actual_grad.detach().cpu().numpy())]))
