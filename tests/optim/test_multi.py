from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim
import pyro.poutine as poutine
from pyro.optim.multi import MixedMultiOptimizer, Newton2d, PyroMultiOptimizer, TorchMultiOptimizer
from tests.common import assert_equal

FACTORIES = [
    lambda: PyroMultiOptimizer(pyro.optim.Adam({'lr': 0.05})),
    lambda: TorchMultiOptimizer(torch.optim.Adam, {'lr': 0.05}),
    lambda: Newton2d(trust_radii={'z': 0.2}),
    lambda: MixedMultiOptimizer([(['y'], PyroMultiOptimizer(pyro.optim.Adam({'lr': 0.05}))),
                                 (['x', 'z'], Newton2d())]),
]


@pytest.mark.parametrize('factory', FACTORIES)
def test_optimizers(factory):
    optim = factory()

    def model(loc, cov):
        x = pyro.param("x", torch.randn(2))
        y = pyro.param("y", torch.randn(3, 2))
        z = pyro.param("z", torch.randn(4, 2).abs(), constraint=constraints.greater_than(-1))
        pyro.sample("obs_x", dist.MultivariateNormal(loc, cov), obs=x)
        with pyro.iarange("y_iarange", 3):
            pyro.sample("obs_y", dist.MultivariateNormal(loc, cov), obs=y)
        with pyro.iarange("z_iarange", 4):
            pyro.sample("obs_z", dist.MultivariateNormal(loc, cov), obs=z)

    loc = torch.tensor([-0.5, 0.5])
    cov = torch.tensor([[1.0, 0.09], [0.09, 0.1]])
    for step in range(200):
        tr = poutine.trace(model).get_trace(loc, cov)
        loss = -tr.log_prob_sum()
        params = {name: pyro.param(name).unconstrained() for name in ["x", "y", "z"]}
        optim.step(loss, params)

    for name in ["x", "y", "z"]:
        actual = pyro.param(name)
        expected = loc.expand(actual.shape)
        assert_equal(actual, expected, prec=1e-2,
                     msg='{} in correct: {} vs {}'.format(name, actual, expected))
