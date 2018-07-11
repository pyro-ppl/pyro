import torch
import numpy as np
import pytest

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.contrib.oed.eig import vi_ape, guide_entropy
from tests.common import assert_equal


def mean_field_guide(batch_tensor, design):
    # A batched variable
    w_p = pyro.param("w_p", 0.2*torch.ones(batch_tensor.shape))
    u_p = pyro.param("u_p", 0.5*torch.ones(batch_tensor.shape))
    pyro.sample("w", dist.Bernoulli(w_p))
    pyro.sample("u", dist.Bernoulli(u_p))


def basic_model(batch_tensor, design):
    pyro.sample("w", dist.Bernoulli(design[0]*torch.ones(batch_tensor.shape)))
    pyro.sample("u", dist.Bernoulli(design[1]*torch.ones(batch_tensor.shape)))
    pyro.sample("y", dist.Delta(torch.Tensor([1.])))


def h(p):
    if p == 0. or p == 1.:
        return 0.
    return -p*np.log(p) - (1-p)*np.log(1-p)


@pytest.mark.parametrize("guide,args,expected_entropy", [
    (mean_field_guide, (torch.Tensor([0.]), None), torch.Tensor([h(0.2) + h(0.5)])),
    (mean_field_guide, (torch.eye(2), None), (h(0.2) + h(0.5))*torch.ones(2, 2))
])
def test_guide_entropy(guide, args, expected_entropy):
    assert_equal(guide_entropy(guide, *args), expected_entropy)


@pytest.mark.parametrize("model,arg,design,guide,expected_ape,n_steps", [
    # Test without running any steps- should get entropy at initialization value
    (basic_model, torch.Tensor([0.0]), torch.Tensor([0.3, 0.4]),
     mean_field_guide, torch.Tensor([h(0.2)+h(0.5)]), 0),
    # These two learn the design (50 steps sufficient for SVI)
    (basic_model, torch.Tensor([0.0]), torch.Tensor([0.3, 0.4]),
     mean_field_guide, torch.Tensor([h(0.3)+h(0.4)]), 50),
    (basic_model, torch.Tensor([0.0]), torch.Tensor([0.3, 0.5]),
     mean_field_guide, torch.Tensor([h(0.3)+h(0.5)]), 50)
])
def test_ape_svi(model, arg, design, guide, expected_ape, n_steps):
    # Reset seed: deals with noise in SVI etc
    pyro.set_rng_seed(42)
    vi_parameters = {
        "guide": lambda d: guide(arg, d),
        # Note: exact details irrelevant, taking 0 steps
        "optim": optim.Adam({"lr": 0.01}),
        "num_steps": n_steps
    }
    is_parameters = {"num_samples": 1}
    ape = vi_ape(lambda d: model(arg, d), design, "y", vi_parameters,
                 is_parameters)
    assert_equal(ape, expected_ape, prec=1e-4)
