# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim
import pyro.poutine as poutine
from pyro.optim.multi import MixedMultiOptimizer, Newton, PyroMultiOptimizer, TorchMultiOptimizer
from tests.common import assert_equal

FACTORIES = [
    lambda: PyroMultiOptimizer(pyro.optim.Adam({'lr': 0.05})),
    lambda: TorchMultiOptimizer(torch.optim.Adam, {'lr': 0.05}),
    lambda: Newton(trust_radii={'z': 0.2}),
    lambda: MixedMultiOptimizer([(['y'], PyroMultiOptimizer(pyro.optim.Adam({'lr': 0.05}))),
                                 (['x', 'z'], Newton())]),
    lambda: MixedMultiOptimizer([(['y'], pyro.optim.Adam({'lr': 0.05})),
                                 (['x', 'z'], Newton())]),
]


@pytest.mark.parametrize('factory', FACTORIES)
def test_optimizers(factory):
    optim = factory()

    def model(loc, cov):
        x = pyro.param("x", torch.randn(2))
        y = pyro.param("y", torch.randn(3, 2))
        z = pyro.param("z", torch.randn(4, 2).abs(), constraint=constraints.greater_than(-1))
        pyro.sample("obs_x", dist.MultivariateNormal(loc, cov), obs=x)
        with pyro.plate("y_plate", 3):
            pyro.sample("obs_y", dist.MultivariateNormal(loc, cov), obs=y)
        with pyro.plate("z_plate", 4):
            pyro.sample("obs_z", dist.MultivariateNormal(loc, cov), obs=z)

    loc = torch.tensor([-0.5, 0.5])
    cov = torch.tensor([[1.0, 0.09], [0.09, 0.1]])
    for step in range(200):
        tr = poutine.trace(model).get_trace(loc, cov)
        loss = -tr.log_prob_sum()
        params = {name: site['value'].unconstrained()
                  for name, site in tr.nodes.items()
                  if site['type'] == 'param'}
        optim.step(loss, params)

    for name in ["x", "y", "z"]:
        actual = pyro.param(name)
        expected = loc.expand(actual.shape)
        assert_equal(actual, expected, prec=1e-2,
                     msg='{} in correct: {} vs {}'.format(name, actual, expected))


def test_multi_optimizer_disjoint_ok():
    parts = [(['w', 'x'], pyro.optim.Adam({'lr': 0.1})),
             (['y', 'z'], pyro.optim.Adam({'lr': 0.01}))]
    MixedMultiOptimizer(parts)


def test_multi_optimizer_overlap_error():
    parts = [(['x', 'y'], pyro.optim.Adam({'lr': 0.1})),
             (['y', 'z'], pyro.optim.Adam({'lr': 0.01}))]
    with pytest.raises(ValueError):
        MixedMultiOptimizer(parts)
