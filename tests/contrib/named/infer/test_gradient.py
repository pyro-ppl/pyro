# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from functorch.dim import dims

import pyro
import pyro.distributions as dist
from pyro.contrib.named.infer import Trace_ELBO
from pyro.distributions.testing import fakes
from pyro.infer import SVI
from pyro.optim import Adam
from tests.common import assert_equal


@pytest.mark.parametrize(
    "reparameterized", [True, False], ids=["reparam", "nonreparam"]
)
def test_plate_elbo_vectorized_particles(reparameterized):
    pyro.enable_validation(False)
    pyro.clear_param_store()
    data = torch.tensor([-0.5, 2.0])
    num_particles = 200000
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal
    i = dims()

    def model():
        data_plate = pyro.plate("data", len(data), dim=i)

        pyro.sample("nuisance_a", Normal(0, 1))
        with data_plate:
            z = pyro.sample("z", Normal(0, 1))
        pyro.sample("nuisance_b", Normal(2, 3))
        with data_plate as idx:
            pyro.sample("x", Normal(z, torch.ones(len(data))[idx]), obs=data[idx])
        pyro.sample("nuisance_c", Normal(4, 5))

    def guide():
        loc = pyro.param("loc", torch.zeros(len(data)))
        scale = pyro.param("scale", torch.ones(len(data)))

        pyro.sample("nuisance_c", Normal(4, 5))
        with pyro.plate("data", len(data), dim=i) as idx:
            pyro.sample("z", Normal(loc[idx], scale[idx]))
        pyro.sample("nuisance_b", Normal(2, 3))
        pyro.sample("nuisance_a", Normal(0, 1))

    optim = Adam({"lr": 0.1})
    loss = Trace_ELBO(
        num_particles=num_particles,
        vectorize_particles=True,
    )
    inference = SVI(model, guide, optim, loss=loss)
    inference.loss_and_grads(model, guide)
    params = dict(pyro.get_param_store().named_parameters())
    actual_grads = {name: param.grad.detach() for name, param in params.items()}

    expected_grads = {
        "loc": torch.tensor([0.5, -2.0]),
        "scale": torch.tensor([1.0, 1.0]),
    }
    assert_equal(actual_grads, expected_grads, prec=0.06)
