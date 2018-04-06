from __future__ import absolute_import, division, print_function

from collections import defaultdict, namedtuple
import logging

import pytest
import torch

import pyro
from pyro.contrib.gp.kernels import Matern32, RBF, WhiteNoise
from pyro.contrib.gp.likelihoods import Gaussian
from pyro.contrib.gp.models import (GPRegression, SparseGPRegression,
                                    VariationalGP, SparseVariationalGP)
import pyro.distributions as dist
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.mcmc import MCMC
from pyro.infer.svi import SVI
import pyro.optim as optim
from tests.common import assert_equal

logging.basicConfig(format='%(levelname)s %(message)s')
logger = logging.getLogger('pyro')
logger.setLevel(logging.INFO)

T = namedtuple("TestGPModel", ["model_class", "X", "y", "kernel", "likelihood"])

X = torch.tensor([[1., 5., 3.], [4., 3., 7.]])
y1D = torch.tensor([2., 1.])
y2D = torch.tensor([[1., 2.], [3., 3.], [1., 4.], [-1., 1.]])
kernel = RBF(input_dim=3, variance=torch.tensor(3.), lengthscale=torch.tensor(2.))
noise = torch.tensor(1e-6)
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
def test_model(model_class, X, y, kernel, likelihood):
    if model_class is SparseGPRegression or model_class is SparseVariationalGP:
        gp = model_class(X, None, kernel, X, likelihood)
    else:
        gp = model_class(X, None, kernel, likelihood)

    loc, var = gp.model()
    if model_class is VariationalGP or model_class is SparseVariationalGP:
        assert_equal(loc.norm().item(), 0)
        assert_equal(var, torch.ones(var.shape[-1]).expand(var.shape))
    else:
        assert_equal(loc.norm().item(), 0)
        assert_equal(var, kernel(X).diag())


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_forward(model_class, X, y, kernel, likelihood):
    if model_class is SparseGPRegression or model_class is SparseVariationalGP:
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
    if model_class is VariationalGP or model_class is SparseVariationalGP:
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


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
def test_forward_with_empty_latent_shape(model_class, X, y, kernel, likelihood):
    # regression models don't use latent_shape, no need for test
    if model_class is GPRegression or model_class is SparseGPRegression:
        return
    elif model_class is VariationalGP:
        gp = model_class(X, y, kernel, likelihood, latent_shape=torch.Size([]))
    else:  # model_class is SparseVariationalGP
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


@pytest.mark.parametrize("model_class, X, y, kernel, likelihood", TEST_CASES, ids=TEST_IDS)
@pytest.mark.init(rng_seed=0)
def test_inference(model_class, X, y, kernel, likelihood):
    # skip variational GP models because variance/lengthscale highly
    # depend on variational parameters
    if model_class is VariationalGP or model_class is SparseVariationalGP:
        return
    elif model_class is GPRegression:
        gp = model_class(X, y, RBF(input_dim=3), likelihood)
    else:  # model_class is SparseGPRegression
        gp = model_class(X, y, RBF(input_dim=3), X, likelihood)
        # fix inducing points because variance/lengthscale highly depend on it
        gp.fix_param("Xu")

    generator = dist.MultivariateNormal(torch.zeros(X.shape[0]), kernel(X))
    target_y = generator(sample_shape=torch.Size([1000])).detach()
    gp.set_data(X, target_y)

    gp.optimize(optim.Adam({"lr": 0.01}), num_steps=1000)

    variance = gp.kernel.get_param("variance")
    lengthscale = gp.kernel.get_param("lengthscale")
    target_variance = kernel.get_param("variance")
    target_lengthscale = kernel.get_param("lengthscale")
    assert_equal(variance, target_variance, prec=0.2)
    assert_equal(lengthscale, target_lengthscale, prec=0.2)


@pytest.mark.init(rng_seed=0)
def test_inference_sgpr():
    N = 1000
    X = dist.Uniform(torch.zeros(N), torch.ones(N)*5).sample()
    y = 0.5 * torch.sin(3*X) + dist.Normal(torch.zeros(N), torch.ones(N)*0.5).sample()
    kernel = RBF(input_dim=1)
    Xu = torch.linspace(0, 5, 10)

    sgpr = SparseGPRegression(X, y, kernel, Xu)
    sgpr.optimize(optim.Adam({"lr": 0.01}), num_steps=1000)

    Xnew = torch.linspace(0, 5, 100)
    loc, var = sgpr(Xnew, full_cov=False)
    target = 0.5 * torch.sin(3*Xnew)

    assert_equal((loc - target).abs().mean().item(), 0, prec=0.05)


@pytest.mark.init(rng_seed=0)
def test_inference_svgp():
    N = 1000
    X = dist.Uniform(torch.zeros(N), torch.ones(N)*5).sample()
    y = 0.5 * torch.sin(3*X) + dist.Normal(torch.zeros(N), torch.ones(N)*0.5).sample()
    kernel = RBF(input_dim=1)
    Xu = torch.linspace(0, 5, 10)

    svgp = SparseVariationalGP(X, y, kernel, Xu, Gaussian())
    svgp.optimize(optim.Adam({"lr": 0.01}), num_steps=1000)

    Xnew = torch.linspace(0, 5, 100)
    loc, var = svgp(Xnew, full_cov=False)
    target = 0.5 * torch.sin(3*Xnew)

    assert_equal((loc - target).abs().mean().item(), 0, prec=0.05)


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

    kernel.set_prior("variance", dist.Uniform(torch.tensor(0.5), torch.tensor(1.5)))
    kernel.set_prior("lengthscale", dist.Uniform(torch.tensor(1.0), torch.tensor(3.0)))

    hmc_kernel = HMC(gp.model, step_size=1)
    mcmc_run = MCMC(hmc_kernel, num_samples=10)

    post_trace = defaultdict(list)
    for trace, _ in mcmc_run._traces():
        variance_name = pyro.param_with_module_name(kernel.name, "variance")
        post_trace["variance"].append(trace.nodes[variance_name]["value"])
        lengthscale_name = pyro.param_with_module_name(kernel.name, "lengthscale")
        post_trace["lengthscale"].append(trace.nodes[lengthscale_name]["value"])
        if model_class is VariationalGP:
            f_name = pyro.param_with_module_name(gp.name, "f")
            post_trace["f"].append(trace.nodes[f_name]["value"])
        if model_class is SparseVariationalGP:
            u_name = pyro.param_with_module_name(gp.name, "u")
            post_trace["u"].append(trace.nodes[u_name]["value"])

    for param in post_trace:
        param_mean = torch.mean(torch.stack(post_trace[param]), 0)
        logger.info("Posterior mean - {}".format(param))
        logger.info(param_mean)


def test_inference_deepGP():
    gp1 = GPRegression(X, None, kernel, name="GPR1")
    Z, _ = gp1.model()
    gp2 = SparseVariationalGP(Z, y2D, Matern32(input_dim=3), Z.clone(),
                              likelihood, name="GPR2")

    def model():
        Z, _ = gp1.model()
        gp2.set_data(Z, y2D)
        gp2.model()

    def guide():
        gp1.guide()
        gp2.guide()

    svi = SVI(model, guide, optim.Adam({}), "ELBO")
    svi.step()
