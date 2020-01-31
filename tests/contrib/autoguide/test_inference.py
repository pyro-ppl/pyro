# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
from functools import partial

import numpy as np
import pytest
import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.distributions.transforms import iterated, block_autoregressive
from pyro.infer.autoguide import (AutoDiagonalNormal, AutoIAFNormal, AutoLaplaceApproximation,
                                  AutoLowRankMultivariateNormal, AutoMultivariateNormal)
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.infer.autoguide.guides import AutoNormalizingFlow
from tests.common import assert_equal
from tests.integration_tests.test_conjugate_gaussian_models import GaussianChain

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.stage("integration", "integration_batch_2")


# conjugate model to test AutoGuide logic from end-to-end (this has a non-mean-field posterior)
class AutoGaussianChain(GaussianChain):

    # this is gross but we need to convert between different posterior factorizations
    def compute_target(self, N):
        self.target_auto_mus = torch.zeros(N + 1)
        self.target_auto_diag_cov = torch.zeros(N + 1)
        self.target_auto_mus[-1] = self.target_mus[N].item()
        self.target_auto_diag_cov[-1] = 1.0 / self.lambda_posts[-1].item()
        for n in range(N - 1, 0, -1):
            self.target_auto_mus[n] += self.target_mus[n].item()
            self.target_auto_mus[n] += self.target_kappas[n].item() * self.target_auto_mus[n + 1]
            self.target_auto_diag_cov[n] += 1.0 / self.lambda_posts[n].item()
            self.target_auto_diag_cov[n] += (self.target_kappas[n].item() ** 2) * self.target_auto_diag_cov[n + 1]

    def test_multivariatate_normal_auto(self):
        self.do_test_auto(3, reparameterized=True, n_steps=10001)

    def do_test_auto(self, N, reparameterized, n_steps):
        logger.debug("\nGoing to do AutoGaussianChain test...")
        pyro.clear_param_store()
        self.setUp()
        self.setup_chain(N)
        self.compute_target(N)
        self.guide = AutoMultivariateNormal(self.model)
        logger.debug("target auto_loc: {}"
                     .format(self.target_auto_mus[1:].detach().cpu().numpy()))
        logger.debug("target auto_diag_cov: {}"
                     .format(self.target_auto_diag_cov[1:].detach().cpu().numpy()))

        # TODO speed up with parallel num_particles > 1
        adam = optim.Adam({"lr": .001, "betas": (0.95, 0.999)})
        svi = SVI(self.model, self.guide, adam, loss=Trace_ELBO())

        for k in range(n_steps):
            loss = svi.step(reparameterized)
            assert np.isfinite(loss), loss

            if k % 1000 == 0 and k > 0 or k == n_steps - 1:
                logger.debug("[step {}] guide mean parameter: {}"
                             .format(k, self.guide.loc.detach().cpu().numpy()))
                L = self.guide.scale_tril
                diag_cov = torch.mm(L, L.t()).diag()
                logger.debug("[step {}] auto_diag_cov: {}"
                             .format(k, diag_cov.detach().cpu().numpy()))

        assert_equal(self.guide.loc.detach(), self.target_auto_mus[1:], prec=0.05,
                     msg="guide mean off")
        assert_equal(diag_cov, self.target_auto_diag_cov[1:], prec=0.07,
                     msg="guide covariance off")


@pytest.mark.parametrize('auto_class', [AutoDiagonalNormal, AutoMultivariateNormal,
                                        AutoLowRankMultivariateNormal, AutoLaplaceApproximation])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceMeanField_ELBO])
def test_auto_diagonal_gaussians(auto_class, Elbo):
    n_steps = 3501 if auto_class == AutoDiagonalNormal else 6001

    def model():
        pyro.sample("x", dist.Normal(-0.2, 1.2))
        pyro.sample("y", dist.Normal(0.2, 0.7))

    if auto_class is AutoLowRankMultivariateNormal:
        guide = auto_class(model, rank=1)
    else:
        guide = auto_class(model)
    adam = optim.Adam({"lr": .001, "betas": (0.95, 0.999)})
    svi = SVI(model, guide, adam, loss=Elbo())

    for k in range(n_steps):
        loss = svi.step()
        assert np.isfinite(loss), loss

    if auto_class is AutoLaplaceApproximation:
        guide = guide.laplace_approximation()

    loc, scale = guide._loc_scale()

    expected_loc = torch.tensor([-0.2, 0.2])
    assert_equal(loc.detach(), expected_loc, prec=0.05,
                 msg="\n".join(["Incorrect guide loc. Expected:",
                                str(expected_loc.cpu().numpy()),
                                "Actual:",
                                str(loc.detach().cpu().numpy())]))
    expected_scale = torch.tensor([1.2, 0.7])
    assert_equal(scale.detach(), expected_scale, prec=0.08,
                 msg="\n".join(["Incorrect guide scale. Expected:",
                                str(expected_scale.cpu().numpy()),
                                "Actual:",
                                str(scale.detach().cpu().numpy())]))


@pytest.mark.parametrize('auto_class', [AutoDiagonalNormal, AutoMultivariateNormal,
                                        AutoLowRankMultivariateNormal, AutoLaplaceApproximation])
def test_auto_transform(auto_class):
    n_steps = 3500

    def model():
        pyro.sample("x", dist.LogNormal(0.2, 0.7))

    if auto_class is AutoLowRankMultivariateNormal:
        guide = auto_class(model, rank=1)
    else:
        guide = auto_class(model)
    adam = optim.Adam({"lr": .001, "betas": (0.90, 0.999)})
    svi = SVI(model, guide, adam, loss=Trace_ELBO())

    for k in range(n_steps):
        loss = svi.step()
        assert np.isfinite(loss), loss

    if auto_class is AutoLaplaceApproximation:
        guide = guide.laplace_approximation()

    loc, scale = guide._loc_scale()
    assert_equal(loc.detach(), torch.tensor([0.2]), prec=0.04,
                 msg="guide mean off")
    assert_equal(scale.detach(), torch.tensor([0.7]), prec=0.04,
                 msg="guide covariance off")


@pytest.mark.parametrize('auto_class', [
    AutoDiagonalNormal,
    AutoIAFNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal,
    AutoLaplaceApproximation,
    lambda m: AutoNormalizingFlow(m, partial(iterated, 2, block_autoregressive)),
])
@pytest.mark.parametrize('Elbo', [Trace_ELBO, TraceMeanField_ELBO])
def test_auto_dirichlet(auto_class, Elbo):
    num_steps = 2000
    prior = torch.tensor([0.5, 1.0, 1.5, 3.0])
    data = torch.tensor([0] * 4 + [1] * 2 + [2] * 5).long()
    posterior = torch.tensor([4.5, 3.0, 6.5, 3.0])

    def model(data):
        p = pyro.sample("p", dist.Dirichlet(prior))
        with pyro.plate("data_plate"):
            pyro.sample("data", dist.Categorical(p).expand_by(data.shape), obs=data)

    guide = auto_class(model)
    svi = SVI(model, guide, optim.Adam({"lr": .003}), loss=Elbo())

    for _ in range(num_steps):
        loss = svi.step(data)
        assert np.isfinite(loss), loss

    expected_mean = posterior / posterior.sum()
    if isinstance(guide, (AutoIAFNormal, AutoNormalizingFlow)):
        loc = guide.transform(torch.zeros(guide.latent_dim))
    else:
        loc = guide.loc
    actual_mean = biject_to(constraints.simplex)(loc)
    assert_equal(actual_mean, expected_mean, prec=0.2, msg=''.join([
        '\nexpected {}'.format(expected_mean.detach().cpu().numpy()),
        '\n  actual {}'.format(actual_mean.detach().cpu().numpy())]))
