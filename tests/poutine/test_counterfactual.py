import logging

import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from tests.common import assert_equal, assert_close

logger = logging.getLogger(__name__)


def test_decorator_interface_do():

    sites = ["x", "y", "z", "_INPUT", "_RETURN"]
    data = {"x": torch.ones(1)}

    @poutine.do(data=data)
    def model():
        p = torch.tensor([0.5])
        loc = torch.zeros(1)
        scale = torch.ones(1)

        x = pyro.sample("x", dist.Normal(loc, scale))
        y = pyro.sample("y", dist.Bernoulli(p))
        z = pyro.sample("z", dist.Normal(loc, scale))
        return dict(x=x, y=y, z=z)

    tr = poutine.trace(model).get_trace()
    for name in sites:
        if name not in data:
            assert name in tr
        else:
            assert name not in tr
            assert_equal(tr.nodes["_RETURN"]["value"][name], data[name])


def test_do_propagation():
    pyro.clear_param_store()

    def model():
        z = pyro.sample("z", dist.Normal(10.0 * torch.ones(1), 0.0001 * torch.ones(1)))
        latent_prob = torch.exp(z) / (torch.exp(z) + torch.ones(1))
        flip = pyro.sample("flip", dist.Bernoulli(latent_prob))
        return flip

    sample_from_model = model()
    z_data = {"z": -10.0 * torch.ones(1)}
    # under model flip = 1 with high probability; so do indirect DO surgery to make flip = 0
    sample_from_do_model = poutine.trace(poutine.do(model, data=z_data))()

    assert_close(sample_from_model, torch.ones(1))
    assert_close(sample_from_do_model, torch.zeros(1))
