import pytest
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.infer
import pyro.optim
from tests.common import assert_equal, assert_not_equal


def model(y1=0, y2=0):
    x = pyro.sample("x", dist.Normal(torch.tensor(0.), torch.tensor(5**0.5)))
    y1 = pyro.sample("y1", dist.Normal(x, torch.tensor(2**0.5)), obs=y1)
    y2 = pyro.sample("y2", dist.Normal(x, torch.tensor(2**0.5)), obs=y2)
    return x


class Guide(nn.Module):
    def __init__(self):
        super(Guide, self).__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        self.std = torch.nn.Parameter(torch.tensor(1.))

    def forward(self, y1, y2):
        mean = self.linear((y1+y2).view(1, 1))[0, 0]
        pyro.sample("x", dist.Normal(mean, self.std))


@pytest.mark.init(rng_seed=7)
def test_csis_sampling():
    guide = Guide()
    csis = pyro.infer.CSIS(model,
                           guide,
                           torch.optim.Adam(guide.parameters()),
                           num_inference_samples=100)
    # observations chosen so that proposal distribution and true posterior will both have zero mean
    posterior = csis.run(y1=torch.tensor(-1.0),
                         y2=torch.tensor(1.0))
    assert_equal(len(posterior.exec_traces), 100)
    marginal = pyro.infer.EmpiricalMarginal(posterior, "x")
    assert_equal(marginal.mean, torch.tensor(0.0), prec=0.25)


@pytest.mark.init(rng_seed=7)
def test_csis_parameter_update():
    guide = Guide()
    initial_parameters = {k: v.item() for k, v in guide.named_parameters()}
    csis = pyro.infer.CSIS(model,
                           guide,
                           torch.optim.Adam(guide.parameters()))
    csis.step()
    updated_parameters = {k: v.item() for k, v in guide.named_parameters()}
    for k, init_v in initial_parameters.items():
        assert_not_equal(init_v, updated_parameters[k])


@pytest.mark.init(rng_seed=7)
def test_csis_validation_batch():
    guide = Guide()
    csis = pyro.infer.CSIS(model,
                           guide,
                           torch.optim.Adam(guide.parameters()))
    csis.set_validation_batch(5)
    init_loss_1 = csis.validation_loss()
    init_loss_2 = csis.validation_loss()
    csis.step()
    next_loss = csis.validation_loss()
    assert_equal(init_loss_1, init_loss_2)
    assert_not_equal(init_loss_1, next_loss)
