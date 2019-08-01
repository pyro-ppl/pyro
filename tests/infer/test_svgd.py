import pytest
import torch

import pyro
import pyro.distributions as dist

from pyro.infer import SVGD, SVGDRBFKernel
from pyro.optim import Adam

from torch.distributions.transforms import AffineTransform


@pytest.mark.parametrize("latent_dist", [#dist.TransformedDistribution(dist.Normal(torch.zeros(1), torch.ones(1)),
                                         #     AffineTransform(loc=-1., scale=2.0))])
    #dist.Normal(torch.zeros(1) + 0.5, torch.ones(1)).to_event(1)])#,
                                         dist.LogNormal(torch.tensor([-1.0]), torch.tensor([0.5])).to_event(1)])
                                         #dist.Gamma(torch.tensor([2.0]), torch.tensor([2.0])).to_event(1)])
                                         #dist.Beta(torch.tensor([0.5]), torch.tensor([0.5])).to_event(1)])
def test_mean_variance(latent_dist):
    pyro.clear_param_store()
    print()

    def model():
        pyro.sample("z", latent_dist)

    kernel = SVGDRBFKernel()
    adam = Adam({"lr": 0.1})
    svgd = SVGD(model, kernel, adam, 1000, 0)

    bandwidth_start = 1.0
    bandwidth_end = 1.0
    n_steps = 301

    for step in range(n_steps):
        bandwidth = bandwidth_start + (step / n_steps) * (bandwidth_end - bandwidth_start)
        svgd.step(bandwidth_factor=bandwidth)
        if step % 100 == 0:
            z = pyro.param('svgd_z')
            print("[step %04d] bandwidth: %.2f  z.mean " % (step, bandwidth),
                  z.mean(0).data.cpu().numpy(), " z.var", z.var(0).data.cpu().numpy())

    z = pyro.param('svgd_z').log()
    print("[step %04d] bandwidth: %.2f  unc.mean " % (step, bandwidth),
          z.mean(0).data.cpu().numpy(), " unc.var", z.var(0).data.cpu().numpy())

    print()
    print("expected mean, variance = ", latent_dist.mean, latent_dist.variance)
    print()
    unc = latent_dist.sample(sample_shape=(1000,)).log()
    print("unconstrained z mean, variance = ", unc.mean(0).data.cpu().numpy(), unc.var(0).data.cpu().numpy())


@pytest.mark.parametrize("shape", [(1, 1), (2, 1, 3), (4, 2), (1, 2, 1, 3)])
def test_shapes(shape):
    pyro.clear_param_store()

    def model():
        pyro.sample("z1", dist.Normal(torch.zeros(shape), 1.0).to_event(len(shape)))
        pyro.sample("z2", dist.Normal(torch.zeros(shape + (5,)), 1.0).to_event(1 + len(shape)))

    num_particles = 7
    svgd = SVGD(model, SVGDRBFKernel(), Adam({"lr": 0.001}), num_particles, 0)

    for step in range(2):
        svgd.step()

    particles = svgd.get_named_particles()
    assert particles['z1'].shape == (num_particles,) + shape
    assert particles['z2'].shape == (num_particles,) + shape + (5,)
