from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import pytest
import torch
from torch.distributions import biject_to, constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim
from pyro.infer import SVI, ADVIDiagonalNormal, ADVIMultivariateNormal
from tests.common import assert_equal
from tests.integration_tests.test_conjugate_gaussian_models import GaussianChain

logger = logging.getLogger(__name__)


# conjugate model to test ADVI logic from end-to-end (this has a non-mean-field posterior)
class ADVIGaussianChain(GaussianChain):

    # this is gross but we need to convert between different posterior factorizations
    def compute_target(self, N):
        self.target_advi_mus = torch.zeros(N + 1)
        self.target_advi_diag_cov = torch.zeros(N + 1)
        self.target_advi_mus[-1] = self.target_mus[N].item()
        self.target_advi_diag_cov[-1] = 1.0 / self.lambda_posts[-1].item()
        for n in range(N - 1, 0, -1):
            self.target_advi_mus[n] += self.target_mus[n].item()
            self.target_advi_mus[n] += self.target_kappas[n].item() * self.target_advi_mus[n + 1]
            self.target_advi_diag_cov[n] += 1.0 / self.lambda_posts[n].item()
            self.target_advi_diag_cov[n] += (self.target_kappas[n].item() ** 2) * self.target_advi_diag_cov[n + 1]

    def test_multivariatate_normal_advi(self):
        self.do_test_advi(3, reparameterized=True, n_steps=10001)

    def do_test_advi(self, N, reparameterized, n_steps):
        logger.debug("\nGoing to do ADVIGaussianChain test...")
        pyro.clear_param_store()
        self.setUp()
        self.setup_chain(N)
        self.compute_target(N)
        self.advi = ADVIMultivariateNormal(self.model)
        logger.debug("target advi_loc: {}".format(self.target_advi_mus[1:].detach().numpy()))
        logger.debug("target advi_diag_cov: {}".format(self.target_advi_diag_cov[1:].detach().numpy()))

        # TODO speed up with parallel num_particles > 1
        adam = optim.Adam({"lr": .0005, "betas": (0.95, 0.999)})
        svi = SVI(self.advi.model, self.advi.guide, adam, loss="ELBO", trace_graph=False)

        for k in range(n_steps):
            loss = svi.step(reparameterized)
            assert np.isfinite(loss), loss

            if k % 1000 == 0 and k > 0 or k == n_steps - 1:
                logger.debug("[step {}] advi mean parameter: {}".format(k, pyro.param("advi_loc").detach().numpy()))
                L = pyro.param("advi_scale_tril")
                diag_cov = torch.mm(L, L.t()).diag()
                logger.debug("[step {}] advi_diag_cov: {}".format(k, diag_cov.detach().numpy()))

        assert_equal(pyro.param("advi_loc"), self.target_advi_mus[1:], prec=0.05,
                     msg="advi mean off")
        assert_equal(diag_cov, self.target_advi_diag_cov[1:], prec=0.07,
                     msg="advi covariance off")


@pytest.mark.parametrize('advi_class', [ADVIDiagonalNormal, ADVIMultivariateNormal])
def test_advi_diagonal_gaussians(advi_class):
    n_steps = 3501 if advi_class == ADVIDiagonalNormal else 6001

    def model():
        pyro.sample("x", dist.Normal(-0.2, 1.2))
        pyro.sample("y", dist.Normal(0.2, 0.7))

    advi = advi_class(model)
    adam = optim.Adam({"lr": .001, "betas": (0.95, 0.999)})
    svi = SVI(advi.model, advi.guide, adam, loss="ELBO", trace_graph=False)

    for k in range(n_steps):
        loss = svi.step()
        assert np.isfinite(loss), loss

    if advi_class == ADVIMultivariateNormal:
        L = pyro.param("advi_scale_tril")
        diag_cov = torch.mm(L, L.t()).diag()
    else:
        diag_cov = torch.pow(pyro.param("advi_scale"), 2.0)

    assert_equal(pyro.param("advi_loc"), torch.tensor([-0.2, 0.2]), prec=0.05,
                 msg="advi mean off")
    assert_equal(diag_cov, torch.tensor([1.44, 0.49]), prec=0.08,
                 msg="advi covariance off")


@pytest.mark.parametrize('advi_class', [ADVIDiagonalNormal, ADVIMultivariateNormal])
def test_advi_transform(advi_class):
    n_steps = 3500

    def model():
        pyro.sample("x", dist.LogNormal(0.2, 0.7))

    advi = advi_class(model)
    adam = optim.Adam({"lr": .001, "betas": (0.90, 0.999)})
    svi = SVI(advi.model, advi.guide, adam, loss="ELBO", trace_graph=False)

    for k in range(n_steps):
        loss = svi.step()
        assert np.isfinite(loss), loss

    if advi_class == ADVIMultivariateNormal:
        L = pyro.param("advi_scale_tril")
        diag_cov = torch.mm(L, L.t()).diag()
    else:
        diag_cov = torch.pow(pyro.param("advi_scale"), 2.0)

    assert_equal(pyro.param("advi_loc"), torch.tensor([0.2]), prec=0.04,
                 msg="advi mean off")
    assert_equal(diag_cov, torch.tensor([0.49]), prec=0.04,
                 msg="advi covariance off")


@pytest.mark.parametrize('advi_class', [ADVIDiagonalNormal, ADVIMultivariateNormal])
def test_advi_dirichlet(advi_class):
    num_steps = 2000
    prior = torch.tensor([0.5, 1.0, 1.5, 3.0])
    data = torch.tensor([0] * 4 + [1] * 2 + [2] * 5).long()
    posterior = torch.tensor([4.5, 3.0, 6.5, 3.0])

    def model(data):
        p = pyro.sample("p", dist.Dirichlet(prior))
        with pyro.iarange("data_iarange"):
            pyro.sample("data", dist.Categorical(p).reshape(data.shape), obs=data)

    advi = advi_class(model)
    svi = SVI(advi.model, advi.guide, optim.Adam({"lr": .003}), loss="ELBO")

    for _ in range(num_steps):
        loss = svi.step(data)
        assert np.isfinite(loss), loss

    expected_mean = posterior / posterior.sum()
    actual_mean = biject_to(constraints.simplex)(pyro.param("advi_loc"))
    assert_equal(actual_mean, expected_mean, prec=0.2, msg=''.join([
        '\nexpected {}'.format(expected_mean.detach().cpu().numpy()),
        '\n  actual {}'.format(actual_mean.detach().cpu().numpy())]))
