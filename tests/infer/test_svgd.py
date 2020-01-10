# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist

from pyro.infer import SVGD, RBFSteinKernel, IMQSteinKernel
from pyro.optim import Adam
from pyro.infer.autoguide.utils import _product

from tests.common import assert_equal


@pytest.mark.parametrize("latent_dist", [dist.Normal(torch.zeros(2), torch.ones(2)).to_event(1),
                                         dist.LogNormal(torch.tensor([-1.0]), torch.tensor([0.7])).to_event(1),
                                         dist.LogNormal(torch.tensor(-1.0), torch.tensor(0.7)),
                                         dist.Beta(torch.tensor([0.3]), torch.tensor([0.7])).to_event(1)])
@pytest.mark.parametrize("mode", ["univariate", "multivariate"])
@pytest.mark.parametrize("stein_kernel", [RBFSteinKernel, IMQSteinKernel])
def test_mean_variance(latent_dist, mode, stein_kernel, verbose=True):
    pyro.clear_param_store()

    def model():
        pyro.sample("z", latent_dist)

    kernel = stein_kernel()
    adam = Adam({"lr": 0.05})
    svgd = SVGD(model, kernel, adam, 200, 0, mode=mode)

    bandwidth_start = 1.0
    bandwidth_end = 5.0
    n_steps = 301

    # scramble initial particles
    svgd.step()
    pyro.param('svgd_particles').unconstrained().data *= 1.3
    pyro.param('svgd_particles').unconstrained().data += 0.7

    for step in range(n_steps):
        kernel.bandwidth_factor = bandwidth_start + (step / n_steps) * (bandwidth_end - bandwidth_start)
        squared_gradients = svgd.step()
        if step % 125 == 0:
            print("[step %03d] " % step, squared_gradients)

    final_particles = svgd.get_named_particles()['z']

    if verbose:
        print("[mean]: actual, expected = ", final_particles.mean(0).data.numpy(),
              latent_dist.mean.data.numpy())
        print("[var]: actual, expected = ", final_particles.var(0).data.numpy(),
              latent_dist.variance.data.numpy())

    assert_equal(final_particles.mean(0), latent_dist.mean, prec=0.01)
    prec = 0.05 if mode == 'multivariate' else 0.02
    assert_equal(final_particles.var(0), latent_dist.variance, prec=prec)


@pytest.mark.parametrize("shape", [(1, 1), (2, 1, 3), (4, 2), (1, 2, 1, 3)])
@pytest.mark.parametrize("stein_kernel", [RBFSteinKernel, IMQSteinKernel])
def test_shapes(shape, stein_kernel):
    pyro.clear_param_store()
    shape1, shape2 = (5,) + shape, shape + (6,)

    mean_init1 = torch.arange(_product(shape1)).double().reshape(shape1) / 100.0
    mean_init2 = torch.arange(_product(shape2)).double().reshape(shape2)

    def model():
        pyro.sample("z1", dist.LogNormal(mean_init1, 1.0e-8).to_event(len(shape1)))
        pyro.sample("scalar", dist.Normal(0.0, 1.0))
        pyro.sample("z2", dist.Normal(mean_init2, 1.0e-8).to_event(len(shape2)))

    num_particles = 7
    svgd = SVGD(model, stein_kernel(), Adam({"lr": 0.0}), num_particles, 0)

    for step in range(2):
        svgd.step()

    particles = svgd.get_named_particles()
    assert particles['z1'].shape == (num_particles,) + shape1
    assert particles['z2'].shape == (num_particles,) + shape2

    for particle in range(num_particles):
        assert_equal(particles['z1'][particle, ...], mean_init1.exp(), prec=1.0e-6)
        assert_equal(particles['z2'][particle, ...], mean_init2, prec=1.0e-6)


@pytest.mark.parametrize("mode", ["univariate", "multivariate"])
@pytest.mark.parametrize("stein_kernel", [RBFSteinKernel, IMQSteinKernel])
def test_conjugate(mode, stein_kernel, verbose=False):
    data = torch.tensor([1.0, 2.0, 3.0, 3.0, 5.0]).unsqueeze(-1).expand(5, 3)
    alpha0 = torch.tensor([1.0, 1.8, 2.3])
    beta0 = torch.tensor([2.3, 1.5, 1.2])
    alpha_n = alpha0 + data.sum(0)  # posterior alpha
    beta_n = beta0 + data.size(0)   # posterior beta

    def model():
        with pyro.plate("rates", alpha0.size(0)):
            latent = pyro.sample("latent",
                                 dist.Gamma(alpha0, beta0))
            with pyro.plate("data", data.size(0)):
                pyro.sample("obs", dist.Poisson(latent), obs=data)

    kernel = stein_kernel()
    adam = Adam({"lr": 0.05})
    svgd = SVGD(model, kernel, adam, 200, 2, mode=mode)

    bandwidth_start = 1.0
    bandwidth_end = 5.0
    n_steps = 451

    for step in range(n_steps):
        kernel.bandwidth_factor = bandwidth_start + (step / n_steps) * (bandwidth_end - bandwidth_start)
        squared_gradients = svgd.step()
        if step % 150 == 0:
            print("[step %03d] " % step, squared_gradients)

    final_particles = svgd.get_named_particles()['latent']
    posterior_dist = dist.Gamma(alpha_n, beta_n)

    if verbose:
        print("[mean]: actual, expected = ", final_particles.mean(0).data.numpy(),
              posterior_dist.mean.data.numpy())
        print("[var]: actual, expected = ", final_particles.var(0).data.numpy(),
              posterior_dist.variance.data.numpy())

    assert_equal(final_particles.mean(0)[0], posterior_dist.mean, prec=0.02)
    prec = 0.05 if mode == 'multivariate' else 0.02
    assert_equal(final_particles.var(0)[0], posterior_dist.variance, prec=prec)
