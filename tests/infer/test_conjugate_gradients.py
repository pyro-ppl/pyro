# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pyro
from pyro.infer.tracegraph_elbo import TraceGraph_ELBO
from tests.common import assert_equal
from tests.integration_tests.test_conjugate_gaussian_models import GaussianChain


class ConjugateChainGradientTests(GaussianChain):

    def test_gradients(self):
        for N in [3, 5]:
            for reparameterized in [True, False]:
                self.do_test_gradients(N, reparameterized)

    def do_test_gradients(self, N, reparameterized):
        pyro.clear_param_store()
        self.setup_chain(N)

        elbo = TraceGraph_ELBO(num_particles=100000, vectorize_particles=True, max_plate_nesting=1)
        elbo.loss_and_grads(self.model, self.guide, reparameterized=reparameterized)

        for i in range(1, N + 1):
            for param_prefix in ["loc_q_%d", "log_sig_q_%d", "kappa_q_%d"]:
                if i == N and param_prefix == 'kappa_q_%d':
                    continue
                actual_grad = pyro.param(param_prefix % i).grad
                assert_equal(actual_grad, 0.0 * actual_grad, prec=0.10, msg="".join([
                             "parameter %s%d" % (param_prefix[:-2], i),
                             "\nexpected = zero vector",
                             "\n  actual = {}".format(actual_grad.detach().cpu().numpy())]))
