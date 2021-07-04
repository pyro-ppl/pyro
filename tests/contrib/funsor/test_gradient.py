# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import funsor
import numpy as np
import pytest
import torch
import torch.optim
from torch.distributions import constraints

import pyro.contrib.funsor
from pyro.distributions.testing import fakes
from pyro.infer.util import zero_grads

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

# _PYRO_BACKEND = "contrib.funsor"
_PYRO_BACKEND = "pyro"


@pytest.mark.parametrize("scale", [1.0, 2.0], ids=["unscaled", "scaled"])
@pytest.mark.parametrize(
    "reparameterized,has_rsample",
    # [(False, None)],
    [(False, None), (True, False), (True, True), (False, None)],
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
    elbo = infer.TraceEnum_ELBO(
        max_plate_nesting=1,  # set this to ensure rng agrees across runs
        num_particles=num_particles,
        vectorize_particles=True,
        strict_enumeration_warning=False,
    )
    pyro.set_rng_seed(0)
    if subsample_size == 1:
        elbo.loss_and_grads(model, guide, subsample=torch.tensor([0], dtype=torch.long))
        elbo.loss_and_grads(model, guide, subsample=torch.tensor([1], dtype=torch.long))
    else:
        elbo.loss_and_grads(
            model, guide, subsample=torch.tensor([0, 1], dtype=torch.long)
        )
    params = dict(pyro.get_param_store().named_parameters())
    normalizer = 2 if subsample else 1
    actual_grads = {
        name: param.grad.detach().cpu().numpy() / normalizer
        for name, param in params.items()
    }

    #  actual_grads = {
    #      "loc": loc_grads.mean(0),
    #      "scale": np.array([scale_grads.mean(0)]),
    #  }

    # reparameterization grads
    pyro.set_rng_seed(0)
    guide_tr = handlers.trace(guide).get_trace(
        subsample=torch.tensor([0, 1], dtype=torch.long)
    )
    model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace(
        subsample=torch.tensor([0, 1], dtype=torch.long)
    )
    guide_tr.compute_log_prob()
    model_tr.compute_log_prob()
    x = data
    z = guide_tr.nodes["z"]["value"].data
    mu = pyro.param("loc").data
    sigma = pyro.param("scale").data
    expected_grads = {
        #  "loc": scale * np.array([0.5, -2.0]),
        #  "scale": scale * np.array([2.0]),
        "scale": (z * (z - mu) - (x - z) * (z - mu) - 1).sum() / sigma,
        "loc": -x + 2 * z,
    }

    # score function estimator
    dmu = (z - mu) / sigma ** 2
    dsigma = (z - mu) ** 2 / sigma ** 3 - 1 / sigma
    loss_i = (
        model_tr.nodes["x"]["log_prob"]
        + model_tr.nodes["z"]["log_prob"]
        - guide_tr.nodes["z"]["log_prob"]
    )
    expected_grads = {
        "scale": -(dsigma * loss_i - dsigma).sum().data,
        "loc": -(dmu * loss_i - dmu).data,
    }

    for name in sorted(params):
        logger.info("expected {} = {}".format(name, expected_grads[name]))
        logger.info("actual   {} = {}".format(name, actual_grads[name]))
    assert_equal(actual_grads, expected_grads, prec=precision)
