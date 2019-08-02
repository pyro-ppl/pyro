import pytest
import torch

from functools import partial

import pyro
import pyro.distributions as dist

from pyro.infer import SVGD, RBFSteinKernel
from pyro.optim import Adam

from tests.common import assert_equal


@pytest.mark.parametrize("latent_dist", [dist.Normal(torch.zeros(2), torch.ones(2)).to_event(1),
                                         dist.LogNormal(torch.tensor([-1.0]), torch.tensor([0.7])).to_event(1),
                                         dist.Beta(torch.tensor([0.4]), torch.tensor([0.6])).to_event(1)])
@pytest.mark.parametrize("mode", ["univariate", "multivariate"])
def test_mean_variance(latent_dist, mode, verbose=False):
    pyro.clear_param_store()

    def model():
        pyro.sample("z", latent_dist)

    # let's make sure the callback mechanism works
    def gradient_callback(squared_gradients, step):
        print("[step %03d] mean squared gradients:" % step, squared_gradients)

    kernel = RBFSteinKernel(mode=mode)
    adam = Adam({"lr": 0.05})
    svgd = SVGD(model, kernel, adam, 200, 0)

    bandwidth_start = 1.0
    bandwidth_end = 5.0
    n_steps = 251

    # scramble initial particles
    svgd.step()
    pyro.param('svgd_z').unconstrained().data *= 1.3
    pyro.param('svgd_z').unconstrained().data += 0.7

    for step in range(n_steps):
        bandwidth = bandwidth_start + (step / n_steps) * (bandwidth_end - bandwidth_start)
        callback = partial(gradient_callback, step=step) if step % 125 == 0 else None
        svgd.step(bandwidth_factor=bandwidth, gradient_callback=callback)

    if verbose:
        print("[mean]: actual, expected = ", pyro.param('svgd_z').mean(0).data.numpy(),
              latent_dist.mean.data.numpy())
        print("[var]: actual, expected = ", pyro.param('svgd_z').var(0).data.numpy(),
              latent_dist.variance.data.numpy())

    assert_equal(pyro.param('svgd_z').mean(0), latent_dist.mean, prec=0.01)
    prec = 0.05 if mode == 'multivariate' else 0.02
    assert_equal(pyro.param('svgd_z').var(0), latent_dist.variance, prec=prec)


@pytest.mark.parametrize("shape", [(1, 1), (2, 1, 3), (4, 2), (1, 2, 1, 3)])
def test_shapes(shape):
    pyro.clear_param_store()

    def model():
        pyro.sample("z1", dist.Normal(torch.zeros(shape), 1.0).to_event(len(shape)))
        pyro.sample("z2", dist.LogNormal(torch.zeros(shape + (5,)), 1.0).to_event(1 + len(shape)))

    num_particles = 7
    svgd = SVGD(model, RBFSteinKernel(), Adam({"lr": 0.001}), num_particles, 0)

    for step in range(2):
        svgd.step()

    particles = svgd.get_named_particles()
    assert particles['z1'].shape == (num_particles,) + shape
    assert particles['z2'].shape == (num_particles,) + shape + (5,)
