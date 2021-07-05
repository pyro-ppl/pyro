# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch

from pyro.distributions.testing import fakes
from tests.common import assert_equal

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor

    import pyro.contrib.funsor

    funsor.set_backend("torch")
    from pyroapi import distributions as dist
    from pyroapi import handlers, infer, pyro, pyro_backend
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)

# _PYRO_BACKEND = os.environ.get("TEST_ENUM_PYRO_BACKEND", "contrib.funsor")


@pytest.mark.parametrize(
    "reparameterized,has_rsample",
    [(True, None), (True, False), (True, True), (False, None)],
    ids=["reparam", "reparam-False", "reparam-True", "nonreparam"],
)
@pytest.mark.parametrize(
    "Elbo",
    [
        "Trace_ELBO",
        "TraceEnum_ELBO",
    ],
)
@pytest.mark.parametrize(
    "backend",
    [
        "pyro",
        "contrib.funsor",
    ],
)
def test_particle_gradient(Elbo, reparameterized, has_rsample, backend):
    with pyro_backend(backend):
        pyro.clear_param_store()
        data = torch.tensor([-0.5, 2.0])
        Normal = dist.Normal if reparameterized else fakes.NonreparameterizedNormal

        def model():
            with pyro.plate("data", len(data)) as ind:
                x = data[ind]
                z = pyro.sample("z", Normal(0, 1))
                pyro.sample("x", Normal(z, 1), obs=x)

        def guide():
            scale = pyro.param("scale", lambda: torch.tensor([1.0]))
            with pyro.plate("data", len(data)):
                loc = pyro.param("loc", lambda: torch.zeros(len(data)), event_dim=0)
                z_dist = Normal(loc, scale)
                if has_rsample is not None:
                    z_dist.has_rsample_(has_rsample)
                pyro.sample("z", z_dist)

        elbo = getattr(infer, Elbo)(
            max_plate_nesting=1,  # set this to ensure rng agrees across runs
            num_particles=1,
            strict_enumeration_warning=False,
        )

        # Elbo gradient estimator
        pyro.set_rng_seed(0)
        elbo.loss_and_grads(model, guide)
        params = dict(pyro.get_param_store().named_parameters())
        actual_grads = {
            name: param.grad.detach().cpu() for name, param in params.items()
        }

        # capture sample values and log_probs
        pyro.set_rng_seed(0)
        guide_tr = handlers.trace(guide).get_trace()
        model_tr = handlers.trace(handlers.replay(model, guide_tr)).get_trace()
        guide_tr.compute_log_prob()
        model_tr.compute_log_prob()
        x = data
        z = guide_tr.nodes["z"]["value"].data
        loc = pyro.param("loc").data
        scale = pyro.param("scale").data

        # expected grads
        if reparameterized and has_rsample is not False:
            # pathwise gradient estimator
            expected_grads = {
                "scale": -(-z * (z - loc) + (x - z) * (z - loc) + 1).sum(
                    0, keepdim=True
                )
                / scale,
                "loc": -(-z + (x - z)),
            }
        else:
            # score function gradient estimator
            elbo = (
                model_tr.nodes["x"]["log_prob"].data
                + model_tr.nodes["z"]["log_prob"].data
                - guide_tr.nodes["z"]["log_prob"].data
            )
            dlogq_dloc = (z - loc) / scale ** 2
            dlogq_dscale = (z - loc) ** 2 / scale ** 3 - 1 / scale
            if Elbo == "TraceEnum_ELBO":
                expected_grads = {
                    "scale": -(dlogq_dscale * elbo - dlogq_dscale).sum(0, keepdim=True),
                    "loc": -(dlogq_dloc * elbo - dlogq_dloc),
                }
            elif Elbo == "Trace_ELBO":
                # expected value of dlogq_dscale and dlogq_dloc is zero
                expected_grads = {
                    "scale": -(dlogq_dscale * elbo).sum(0, keepdim=True),
                    "loc": -(dlogq_dloc * elbo),
                }

        for name in sorted(params):
            logger.info("expected {} = {}".format(name, expected_grads[name]))
            logger.info("actual   {} = {}".format(name, actual_grads[name]))

        assert_equal(actual_grads, expected_grads, prec=1e-4)
