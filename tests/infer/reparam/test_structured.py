# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import MCMC, NUTS, SVI, JitTrace_ELBO, Trace_ELBO
from pyro.infer.autoguide import AutoStructured
from pyro.infer.reparam import StructuredReparam

from .util import check_init_reparam


def neals_funnel(dim=10):
    y = pyro.sample("y", dist.Normal(0, 3))
    with pyro.plate("D", dim):
        return pyro.sample("x", dist.Normal(0, torch.exp(y / 2)))


@pytest.mark.parametrize("jit", [False, True])
def test_neals_funnel_smoke(jit):
    dim = 10

    guide = AutoStructured(
        neals_funnel,
        conditionals={"y": "normal", "x": "mvn"},
        dependencies={"x": {"y": "linear"}},
    )
    Elbo = JitTrace_ELBO if jit else Trace_ELBO
    svi = SVI(neals_funnel, guide, optim.Adam({"lr": 1e-10}), Elbo())
    for _ in range(1000):
        try:
            svi.step(dim=dim)
        except SystemError as e:
            if "returned a result with an error set" in str(e):
                pytest.xfail(reason="PyTorch jit bug")
            else:
                raise e from None

    rep = StructuredReparam(guide)
    model = rep.reparam(neals_funnel)
    nuts = NUTS(model, max_tree_depth=3, jit_compile=jit)
    mcmc = MCMC(nuts, num_samples=50, warmup_steps=50)
    mcmc.run(dim)
    samples = mcmc.get_samples()
    # XXX: `MCMC.get_samples` adds a leftmost batch dim to all sites,
    # not uniformly at -max_plate_nesting-1; hence the unsqueeze.
    samples = {k: v.unsqueeze(1) for k, v in samples.items()}
    transformed_samples = rep.transform_samples(samples)
    assert isinstance(transformed_samples, dict)
    assert set(transformed_samples) == {"x", "y"}


def test_init():
    guide = AutoStructured(
        neals_funnel,
        conditionals={"y": "normal", "x": "mvn"},
        dependencies={"x": {"y": "linear"}},
    )
    guide()

    check_init_reparam(neals_funnel, StructuredReparam(guide))
