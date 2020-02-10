# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
from collections import namedtuple

import pytest
import torch

import pyro.distributions as dist
from pyro.contrib.gp.kernels import Cosine, Matern32, RBF, WhiteNoise
from pyro.contrib.gp.likelihoods import Gaussian
from pyro.contrib.gp.models import (GPLVM, GPRegression, SparseGPRegression,
                                    VariationalGP, VariationalSparseGP)
from pyro.contrib.gp.util import train
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.api import MCMC
from pyro.nn.module import PyroSample
from tests.common import assert_equal

logger = logging.getLogger(__name__)

T = namedtuple("TestGPModel", ["model_class", "X", "y", "kernel", "likelihood"])

X = torch.tensor([[1., 5., 3.], [4., 3., 7.]])
y1D = torch.tensor([2., 1.])
y2D = torch.tensor([[1., 2.], [3., 3.], [1., 4.], [-1., 1.]])
noise = torch.tensor(1e-7)


def _kernel():
    return RBF(input_dim=3, variance=torch.tensor(3.), lengthscale=torch.tensor(2.))


def _likelihood():
    return Gaussian(torch.tensor(1e-7))


def _TEST_CASES():
    TEST_CASES = [
        T(
            GPRegression,
            X, y1D, _kernel(), noise
        ),
        T(
            GPRegression,
            X, y2D, _kernel(), noise
        ),
        T(
            SparseGPRegression,
            X, y1D, _kernel(), noise
        ),
        T(
            SparseGPRegression,
            X, y2D, _kernel(), noise
        ),
        T(
            VariationalGP,
            X, y1D, _kernel(), _likelihood()
        ),
        T(
            VariationalGP,
            X, y2D, _kernel(), _likelihood()
        ),
        T(
            VariationalSparseGP,
            X, y1D, _kernel(), _likelihood()
        ),
        T(
            VariationalSparseGP,
            X, y2D, _kernel(), _likelihood()
        ),
    ]

    return TEST_CASES


TEST_IDS = [t[0].__name__ + "_y{}D".format(str(t[2].dim()))
            for t in _TEST_CASES()]


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", _TEST_CASES(), ids=TEST_IDS)
def test_model(model_class, X, y, kernel, likelihood):
    if model_class is SparseGPRegression or model_class is VariationalSparseGP:
        gp = model_class(X, None, kernel, X, likelihood)
    else:
        gp = model_class(X, None, kernel, likelihood)

    loc, var = gp.model()
    if model_class is VariationalGP or model_class is VariationalSparseGP:
        assert_equal(loc.norm().item(), 0)
        assert_equal(var, torch.ones(var.shape[-1]).expand(var.shape))
    else:
        assert_equal(loc.norm().item(), 0)
        assert_equal(var, kernel(X).diag())


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", _TEST_CASES(), ids=TEST_IDS)
def test_forward(model_class, X, y, kernel, likelihood):
    if model_class is SparseGPRegression or model_class is VariationalSparseGP:
        gp = model_class(X, y, kernel, X, likelihood)
    else:
        gp = model_class(X, y, kernel, likelihood)

    # test shape
    Xnew = torch.tensor([[2.0, 3.0, 1.0]])
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

    # test trivial forward: Xnew = X
    loc, cov = gp(X, full_cov=True)
    if model_class is VariationalGP or model_class is VariationalSparseGP:
        assert_equal(loc.norm().item(), 0)
        assert_equal(cov, torch.eye(cov.shape[-1]).expand(cov.shape))
    else:
        assert_equal(loc, y)
        assert_equal(cov.norm().item(), 0)

    # test same input forward: Xnew[0,:] = Xnew[1,:] = ...
    Xnew = torch.tensor([[2.0, 3.0, 1.0]]).expand(10, 3)
    loc, cov = gp(Xnew, full_cov=True)
    loc_diff = loc - loc[..., :1].expand(y.shape[:-1] + (10,))
    assert_equal(loc_diff.norm().item(), 0)
    cov_diff = cov - cov[..., :1, :1].expand(y.shape[:-1] + (10, 10))
    assert_equal(cov_diff.norm().item(), 0)

    # test noise kernel forward: kernel = WhiteNoise
    gp.kernel = WhiteNoise(input_dim=3, variance=torch.tensor(10.))
    loc, cov = gp(X, full_cov=True)
    assert_equal(loc.norm().item(), 0)
    assert_equal(cov, torch.eye(cov.shape[-1]).expand(cov.shape) * 10)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", _TEST_CASES(), ids=TEST_IDS)
def test_forward_with_empty_latent_shape(model_class, X, y, kernel, likelihood):
    # regression models don't use latent_shape, no need for test
    if model_class is GPRegression or model_class is SparseGPRegression:
        return
    elif model_class is VariationalGP:
        gp = model_class(X, y, kernel, likelihood, latent_shape=torch.Size([]))
    else:  # model_class is VariationalSparseGP
        gp = model_class(X, y, kernel, X, likelihood, latent_shape=torch.Size([]))

    # test shape
    Xnew = torch.tensor([[2.0, 3.0, 1.0]])
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


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", _TEST_CASES(), ids=TEST_IDS)
@pytest.mark.init(rng_seed=0)
def test_inference(model_class, X, y, kernel, likelihood):
    # skip variational GP models because variance/lengthscale highly
    # depend on variational parameters
    if model_class is VariationalGP or model_class is VariationalSparseGP:
        return
    elif model_class is GPRegression:
        gp = model_class(X, y, RBF(input_dim=3), likelihood)
    else:  # model_class is SparseGPRegression
        gp = model_class(X, y, RBF(input_dim=3), X, likelihood)
        # fix inducing points because variance/lengthscale highly depend on it
        gp.Xu.requires_grad_(False)

    generator = dist.MultivariateNormal(torch.zeros(X.shape[0]), kernel(X))
    target_y = generator(sample_shape=torch.Size([1000])).detach()
    gp.set_data(X, target_y)

    train(gp)

    y_cov = gp.kernel(X)
    target_y_cov = kernel(X)
    assert_equal(y_cov, target_y_cov, prec=0.15)


@pytest.mark.init(rng_seed=0)
def test_inference_sgpr():
    N = 1000
    X = dist.Uniform(torch.zeros(N), torch.ones(N)*5).sample()
    y = 0.5 * torch.sin(3*X) + dist.Normal(torch.zeros(N), torch.ones(N)*0.5).sample()
    kernel = RBF(input_dim=1)
    Xu = torch.arange(0., 5.5, 0.5)

    sgpr = SparseGPRegression(X, y, kernel, Xu)
    train(sgpr)

    Xnew = torch.arange(0., 5.05, 0.05)
    loc, var = sgpr(Xnew, full_cov=False)
    target = 0.5 * torch.sin(3*Xnew)

    assert_equal((loc - target).abs().mean().item(), 0, prec=0.07)


@pytest.mark.init(rng_seed=0)
def test_inference_vsgp():
    N = 1000
    X = dist.Uniform(torch.zeros(N), torch.ones(N)*5).sample()
    y = 0.5 * torch.sin(3*X) + dist.Normal(torch.zeros(N), torch.ones(N)*0.5).sample()
    kernel = RBF(input_dim=1)
    Xu = torch.arange(0., 5.5, 0.5)

    vsgp = VariationalSparseGP(X, y, kernel, Xu, Gaussian())
    optimizer = torch.optim.Adam(vsgp.parameters(), lr=0.03)
    train(vsgp, optimizer)

    Xnew = torch.arange(0., 5.05, 0.05)
    loc, var = vsgp(Xnew, full_cov=False)
    target = 0.5 * torch.sin(3*Xnew)

    assert_equal((loc - target).abs().mean().item(), 0, prec=0.06)


@pytest.mark.init(rng_seed=0)
def test_inference_whiten_vsgp():
    N = 1000
    X = dist.Uniform(torch.zeros(N), torch.ones(N)*5).sample()
    y = 0.5 * torch.sin(3*X) + dist.Normal(torch.zeros(N), torch.ones(N)*0.5).sample()
    kernel = RBF(input_dim=1)
    Xu = torch.arange(0., 5.5, 0.5)

    vsgp = VariationalSparseGP(X, y, kernel, Xu, Gaussian(), whiten=True)
    train(vsgp)

    Xnew = torch.arange(0., 5.05, 0.05)
    loc, var = vsgp(Xnew, full_cov=False)
    target = 0.5 * torch.sin(3*Xnew)

    assert_equal((loc - target).abs().mean().item(), 0, prec=0.07)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", _TEST_CASES(), ids=TEST_IDS)
def test_inference_with_empty_latent_shape(model_class, X, y, kernel, likelihood):
    # regression models don't use latent_shape (default=torch.Size([]))
    if model_class is GPRegression or model_class is SparseGPRegression:
        return
    elif model_class is VariationalGP:
        gp = model_class(X, y, kernel, likelihood, latent_shape=torch.Size([]))
    else:  # model_class is SparseVariationalGP
        gp = model_class(X, y, kernel, X.clone(), likelihood, latent_shape=torch.Size([]))

    train(gp, num_steps=1)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", _TEST_CASES(), ids=TEST_IDS)
def test_inference_with_whiten(model_class, X, y, kernel, likelihood):
    # regression models don't use whiten
    if model_class is GPRegression or model_class is SparseGPRegression:
        return
    elif model_class is VariationalGP:
        gp = model_class(X, y, kernel, likelihood, whiten=True)
    else:  # model_class is SparseVariationalGP
        gp = model_class(X, y, kernel, X.clone(), likelihood, whiten=True)

    train(gp, num_steps=1)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", _TEST_CASES(), ids=TEST_IDS)
def test_hmc(model_class, X, y, kernel, likelihood):
    if model_class is SparseGPRegression or model_class is VariationalSparseGP:
        gp = model_class(X, y, kernel, X.clone(), likelihood)
    else:
        gp = model_class(X, y, kernel, likelihood)

    kernel.variance = PyroSample(dist.Uniform(torch.tensor(0.5), torch.tensor(1.5)))
    kernel.lengthscale = PyroSample(dist.Uniform(torch.tensor(1.0), torch.tensor(3.0)))

    hmc_kernel = HMC(gp.model, step_size=1)
    mcmc = MCMC(hmc_kernel, num_samples=10)
    mcmc.run()

    for name, param in mcmc.get_samples().items():
        param_mean = torch.mean(param, 0)
        logger.info("Posterior mean - {}".format(name))
        logger.info(param_mean)


def test_inference_deepGP():
    gp1 = GPRegression(X, None, RBF(input_dim=3, variance=torch.tensor(3.),
                                    lengthscale=torch.tensor(2.)))
    Z, _ = gp1.model()
    gp2 = VariationalSparseGP(Z, y2D, Matern32(input_dim=3), Z.clone(),
                              Gaussian(torch.tensor(1e-6)))

    class DeepGP(torch.nn.Module):
        def __init__(self, gp1, gp2):
            super().__init__()
            self.gp1 = gp1
            self.gp2 = gp2

        def model(self):
            Z, _ = self.gp1.model()
            self.gp2.set_data(Z, y2D)
            self.gp2.model()

        def guide(self):
            self.gp1.guide()
            self.gp2.guide()

    deepgp = DeepGP(gp1, gp2)
    train(deepgp, num_steps=1)


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", _TEST_CASES(), ids=TEST_IDS)
def test_gplvm(model_class, X, y, kernel, likelihood):
    if model_class is SparseGPRegression or model_class is VariationalSparseGP:
        gp = model_class(X, y, kernel, X.clone(), likelihood)
    else:
        gp = model_class(X, y, kernel, likelihood)

    gplvm = GPLVM(gp)
    # test inference
    train(gplvm, num_steps=1)
    # test forward
    gplvm(Xnew=X)


def _pre_test_mean_function():
    def f(x):
        return 2 * x + 3 + 5 * torch.sin(7 * x)

    X = torch.arange(100, dtype=torch.Tensor().dtype)
    y = f(X)
    Xnew = torch.arange(100, 150, dtype=torch.Tensor().dtype)
    ynew = f(Xnew)

    kernel = Cosine(input_dim=1)

    class Trend(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.Parameter(torch.tensor(0.))
            self.b = torch.nn.Parameter(torch.tensor(1.))

        def forward(self, x):
            return self.a * x + self.b

    trend = Trend()
    return X, y, Xnew, ynew, kernel, trend


def _mape(y_true, y_pred):
    return ((y_pred - y_true) / y_true).abs().mean()


def _post_test_mean_function(gpmodule, Xnew, y_true):
    assert_equal(gpmodule.mean_function.a.item(), 2, prec=0.03)
    assert_equal(gpmodule.mean_function.b.item(), 3, prec=0.03)

    y_pred, _ = gpmodule(Xnew)
    assert_equal(_mape(y_true, y_pred).item(), 0, prec=0.02)


def test_mean_function_GPR():
    X, y, Xnew, ynew, kernel, mean_fn = _pre_test_mean_function()
    gpmodule = GPRegression(X, y, kernel, mean_function=mean_fn)
    train(gpmodule)
    _post_test_mean_function(gpmodule, Xnew, ynew)


def test_mean_function_SGPR():
    X, y, Xnew, ynew, kernel, mean_fn = _pre_test_mean_function()
    Xu = X[::20].clone()
    gpmodule = SparseGPRegression(X, y, kernel, Xu, mean_function=mean_fn)
    train(gpmodule)
    _post_test_mean_function(gpmodule, Xnew, ynew)


def test_mean_function_SGPR_DTC():
    X, y, Xnew, ynew, kernel, mean_fn = _pre_test_mean_function()
    Xu = X[::20].clone()
    gpmodule = SparseGPRegression(X, y, kernel, Xu, mean_function=mean_fn, approx="DTC")
    train(gpmodule)
    _post_test_mean_function(gpmodule, Xnew, ynew)


def test_mean_function_SGPR_FITC():
    X, y, Xnew, ynew, kernel, mean_fn = _pre_test_mean_function()
    Xu = X[::20].clone()
    gpmodule = SparseGPRegression(X, y, kernel, Xu, mean_function=mean_fn, approx="FITC")
    train(gpmodule)
    _post_test_mean_function(gpmodule, Xnew, ynew)


def test_mean_function_VGP():
    X, y, Xnew, ynew, kernel, mean_fn = _pre_test_mean_function()
    likelihood = Gaussian()
    gpmodule = VariationalGP(X, y, kernel, likelihood, mean_function=mean_fn)
    train(gpmodule)
    _post_test_mean_function(gpmodule, Xnew, ynew)


def test_mean_function_VGP_whiten():
    X, y, Xnew, ynew, kernel, mean_fn = _pre_test_mean_function()
    likelihood = Gaussian()
    gpmodule = VariationalGP(X, y, kernel, likelihood, mean_function=mean_fn,
                             whiten=True)
    optimizer = torch.optim.Adam(gpmodule.parameters(), lr=0.1)
    train(gpmodule, optimizer)
    _post_test_mean_function(gpmodule, Xnew, ynew)


def test_mean_function_VSGP():
    X, y, Xnew, ynew, kernel, mean_fn = _pre_test_mean_function()
    Xu = X[::20].clone()
    likelihood = Gaussian()
    gpmodule = VariationalSparseGP(X, y, kernel, Xu, likelihood, mean_function=mean_fn)
    optimizer = torch.optim.Adam(gpmodule.parameters(), lr=0.02)
    train(gpmodule, optimizer)
    _post_test_mean_function(gpmodule, Xnew, ynew)


def test_mean_function_VSGP_whiten():
    X, y, Xnew, ynew, kernel, mean_fn = _pre_test_mean_function()
    Xu = X[::20].clone()
    likelihood = Gaussian()
    gpmodule = VariationalSparseGP(X, y, kernel, Xu, likelihood, mean_function=mean_fn,
                                   whiten=True)
    optimizer = torch.optim.Adam(gpmodule.parameters(), lr=0.1)
    train(gpmodule, optimizer)
    _post_test_mean_function(gpmodule, Xnew, ynew)
