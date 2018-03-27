from __future__ import absolute_import, division, print_function

from collections import defaultdict, namedtuple
import logging

import pytest
import torch

import pyro
from pyro.contrib.gp.kernels import RBF
from pyro.contrib.gp.likelihoods import Gaussian
from pyro.contrib.gp.models import (GPRegression, SparseGPRegression,
                                    VariationalGP, SparseVariationalGP)
import pyro.distributions as dist
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.mcmc import MCMC
from tests.common import assert_equal

logging.basicConfig(format='%(levelname)s %(message)s')
logger = logging.getLogger('pyro')
logger.setLevel(logging.INFO)

T = namedtuple("TestGPModel", ["model_class", "X", "y", "kernel", "likelihood"])

X = torch.tensor([[1, 5, 3], [4, 3, 7]])
y1D = torch.tensor([2, 1])
y2D = torch.tensor([[1, 2], [3, 3], [1, 4], [-1, 1]])
kernel = RBF(input_dim=3, variance=torch.tensor([1]), lengthscale=torch.tensor([3]))
noise = torch.tensor([1e-6])
likelihood = Gaussian(noise)

TEST_CASES = [
    T(
        GPRegression,
        X, y1D, kernel, noise
    ),
    T(
        GPRegression,
        X, y2D, kernel, noise
    ),
    T(
        SparseGPRegression,
        X, y1D, kernel, noise
    ),
    T(
        SparseGPRegression,
        X, y2D, kernel, noise
    ),
    T(
        VariationalGP,
        X, y1D, kernel, likelihood
    ),
    T(
        VariationalGP,
        X, y2D, kernel, likelihood
    ),
    T(
        SparseVariationalGP,
        X, y1D, kernel, likelihood
    ),
    T(
        SparseVariationalGP,
        X, y2D, kernel, likelihood
    ),
]

TEST_IDS = [t[0].__name__ + "_y{}D".format(str(t[2].dim()))
            for t in TEST_CASES]


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_model_forward(model_class, X, y, kernel, likelihood):
    if model_class is SparseGPRegression or model_class is SparseVariationalGP:
        gp = model_class(X, y, kernel, X, likelihood)
    else:
        gp = model_class(X, y, kernel, likelihood)

    # test shape
    Xnew = torch.tensor([[2, 3, 1]])
    loc0, cov0 = gp(Xnew, full_cov=True)
    loc1, var1 = gp(Xnew, full_cov=False)
    assert loc0.dim() == y.dim()
    assert loc0.shape[-1] == Xnew.shape[0]
    # test latent shape
    assert loc0.shape[:-1] == y.shape[:-1]
    assert cov0.shape[:-2] == y.shape[:-1]
    assert cov0.shape[-1] == cov0.shape[-2]
    assert cov0.shape[-1] == Xnew.shape[0]
    assert_equal(loc0, loc1)
    n = Xnew.shape[0]
    cov0_diag = torch.stack([mat.diag() for mat in cov0.view(-1, n, n)]).reshape(var1.shape)
    assert_equal(cov0_diag, var1)

    # test trivial forward
    # for variational models, inferences depend on variational parameters, so skip
    if model_class is VariationalGP or model_class is SparseVariationalGP:
        pass
    else:
        loc, cov = gp(X, full_cov=True)
        assert_equal(loc, y)
        assert_equal(cov.norm().item(), 0)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_inference(model_class, X, y, kernel, likelihood):
    if model_class is SparseGPRegression or model_class is SparseVariationalGP:
        gp = model_class(X, y, kernel, X, likelihood)
    else:
        gp = model_class(X, y, kernel, likelihood)

    gp.optimize(num_steps=1)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_model_forward_with_empty_latent_shape(model_class, X, y, kernel, likelihood):
    # regression models don't use latent_shape (default=torch.Size([]))
    if model_class is GPRegression or model_class is SparseGPRegression:
        return
    elif model_class is VariationalGP:
        gp = model_class(X, y, kernel, likelihood, latent_shape=torch.Size([]))
    else:  # model_class is SparseVariationalGP
        gp = model_class(X, y, kernel, X, likelihood, latent_shape=torch.Size([]))

    gp.optimize(num_steps=1)

    # test shape
    Xnew = torch.tensor([[2, 3, 1]])
    loc0, cov0 = gp(Xnew, full_cov=True)
    loc1, var1 = gp(Xnew, full_cov=False)
    assert loc0.shape[-1] == Xnew.shape[0]
    assert cov0.shape[-1] == cov0.shape[-2]
    assert cov0.shape[-1] == Xnew.shape[0]
    # test latent shape
    assert loc0.shape[:-1] == torch.Size([])
    assert cov0.shape[:-2] == torch.Size([])
    assert_equal(loc0, loc1)
    assert_equal(cov0.diag(), var1)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_inference_with_empty_latent_shape(model_class, X, y, kernel, likelihood):
    # regression models don't use latent_shape (default=torch.Size([]))
    if model_class is GPRegression or model_class is SparseGPRegression:
        return
    elif model_class is VariationalGP:
        gp = model_class(X, y, kernel, likelihood, latent_shape=torch.Size([]))
    else:  # model_class is SparseVariationalGP
        gp = model_class(X, y, kernel, X, likelihood, latent_shape=torch.Size([]))

    gp.optimize(num_steps=1)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_hmc(model_class, X, y, kernel, likelihood):
    if model_class is SparseGPRegression or model_class is SparseVariationalGP:
        gp = model_class(X, y, kernel, X, likelihood)
    else:
        gp = model_class(X, y, kernel, likelihood)

    kernel.set_prior("variance", dist.Uniform(torch.tensor([0.5]), torch.tensor([1.5])))
    kernel.set_prior("lengthscale", dist.Uniform(torch.tensor([1]), torch.tensor([3])))

    hmc_kernel = HMC(gp.model, step_size=1)
    mcmc_run = MCMC(hmc_kernel, num_samples=10)

    post_trace = defaultdict(list)
    for trace, _ in mcmc_run._traces():
        print("New traceeeeeeeee")
        variance_name = pyro.param_with_module_name(kernel, "variance")
        post_trace["variance"].append(trace.nodes[variance_name]["value"])
        lengthscale_name = pyro.param_with_module_name(kernel, "lengthscale")
        post_trace["lengthscale"].append(trace.nodes[lengthscale_name]["value"])

    variance_mean = torch.mean(torch.stack(post_trace["variance"]), 0)
    logger.info("Posterior mean - {}".format("variance"))
    logger.info(variance_mean)
    lengthscale_mean = torch.mean(torch.stack(post_trace["lengthscale"]), 0)
    logger.info("Posterior mean - {}".format("lengthscale"))
    logger.info(lengthscale_mean)
