# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np
import pytest
import torch
import torch.optim
from torch.distributions import constraints
from pyro.infer.util import zero_grads
from pyro.distributions.testing import fakes

import funsor

import pyro.contrib.funsor

funsor.set_backend("torch")
from pyroapi import distributions as dist
from pyroapi import handlers, infer, pyro, pyro_backend
#  import pyro
#  import pyro.distributions as dist
#  import pyro.poutine as poutine
#  from pyro.distributions.testing import fakes
#  from pyro.infer import (
#      SVI,
#      JitTrace_ELBO,
#      JitTraceEnum_ELBO,
#      JitTraceGraph_ELBO,
#      JitTraceMeanField_ELBO,
#      Trace_ELBO,
#      TraceEnum_ELBO,
#      TraceGraph_ELBO,
#      TraceMeanField_ELBO,
#      config_enumerate,
#  )
from pyro.optim import Adam
from tests.common import assert_equal, xfail_if_not_implemented, xfail_param

logger = logging.getLogger(__name__)

_PYRO_BACKEND = "contrib.funsor"
# _PYRO_BACKEND = "pyro"

def DiffTrace_ELBO(*args, **kwargs):
    return infer.Trace_ELBO(*args, **kwargs).differentiable_loss


@pytest.mark.parametrize("scale", [1.0, 2.0], ids=["unscaled", "scaled"])
@pytest.mark.parametrize(
    "reparameterized,has_rsample",
    [(True, None), (True, False), (True, True), (False, None)],
    ids=["reparam", "reparam-False", "reparam-True", "nonreparam"],
)
@pytest.mark.parametrize("subsample", [False, True], ids=["full", "subsample"])
@pytest.mark.parametrize(
    "Elbo,local_samples",
    [
        (infer.Trace_ELBO, False),
    ],
)
@pyro_backend(_PYRO_BACKEND)
def test_funsor_gradient(
    Elbo, reparameterized, has_rsample, subsample, local_samples, scale
):
    pyro.clear_param_store()
    data = torch.tensor([-0.5, 2.0])
    subsample_size = 1 if subsample else len(data)
    precision = 0.3 * scale
    Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

    def model(subsample):
        with pyro.plate("data", len(data), subsample_size, subsample) as ind:
            x = data[ind]
            z = pyro.sample("z", Normal(0, 1))
            pyro.sample("x", Normal(z, 1), obs=x)

    def guide(subsample):
        scale = pyro.param("scale", lambda: torch.tensor([1.0]))
        with pyro.plate("data", len(data), subsample_size, subsample):
            loc = pyro.param("loc", lambda: torch.zeros(len(data)), event_dim=0)
            z_dist = Normal(loc, scale)
            if has_rsample is not None:
                z_dist.has_rsample_(has_rsample)
            pyro.sample("z", z_dist)

    if scale != 1.0:
        model = handlers.scale(model, scale=scale)
        guide = handlers.scale(guide, scale=scale)

    num_particles = 1
    if local_samples:
        guide = infer.config_enumerate(guide, num_samples=num_particles)
        num_particles = 1

    optim = Adam({"lr": 0.1})
    elbo = Elbo(
        max_plate_nesting=1,  # set this to ensure rng agrees across runs
        num_particles=num_particles,
        vectorize_particles=True,
        strict_enumeration_warning=False,
    )
    inference = infer.SVI(model, guide, optim, loss=elbo)
    loc_grads = np.zeros((1000, 2))
    scale_grads = np.zeros((1000,))
    for i in range(1000):
        if subsample_size == 1:
            inference.loss_and_grads(
                model, guide, subsample=torch.tensor([0], dtype=torch.long)
            )
            inference.loss_and_grads(
                model, guide, subsample=torch.tensor([1], dtype=torch.long)
            )
        else:
            inference.loss_and_grads(
                model, guide, subsample=torch.tensor([0, 1], dtype=torch.long)
            )
        params = dict(pyro.get_param_store().named_parameters())
        normalizer = 2 if subsample else 1
        loc_grads[i, :] = params["loc"].grad.detach().cpu().numpy() / normalizer
        scale_grads[i] = params["scale"].grad.detach().cpu().numpy() / normalizer
        #  actual_grads = {
        #      name: param.grad.detach().cpu().numpy() / normalizer
        #      for name, param in params.items()
        #  }
        #  params = set(
        #      site["value"].unconstrained() for site in param_capture.trace.nodes.values()
        #  )

        # zero gradients
        zero_grads(set(params.values()))

    actual_grads = {
        "loc": loc_grads.mean(0),
        "scale": np.array([scale_grads.mean(0)]),
    }

    expected_grads = {
        "loc": scale * np.array([0.5, -2.0]),
        "scale": scale * np.array([2.0]),
    }
    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)
