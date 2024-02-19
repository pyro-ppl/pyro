# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import pytest
import torch
from torch.autograd import grad
from torch.distributions import constraints

from tests.common import assert_equal

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor

    import pyro.contrib.funsor

    funsor.set_backend("torch")
    from pyroapi import distributions as dist
    from pyroapi import infer, pyro, pyro_backend
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("num_samples", [None, 200])
@pytest.mark.parametrize("max_plate_nesting", [2, 3])
@pytest.mark.parametrize("tmc_strategy", ["diagonal", "mixture"])
def test_tmc_categoricals(depth, max_plate_nesting, num_samples, tmc_strategy):
    def model():
        x = pyro.sample("x0", dist.Categorical(pyro.param("q0")))
        with pyro.plate("local", 3):
            for i in range(1, depth):
                x = pyro.sample(
                    "x{}".format(i),
                    dist.Categorical(pyro.param("q{}".format(i))[..., x, :]),
                )
            with pyro.plate("data", 4):
                pyro.sample("y", dist.Bernoulli(pyro.param("qy")[..., x]), obs=data)

    with pyro_backend("pyro"):
        # initialize
        qs = [pyro.param("q0", torch.tensor([0.4, 0.6], requires_grad=True))]
        for i in range(1, depth):
            qs.append(
                pyro.param(
                    "q{}".format(i),
                    torch.randn(2, 2).abs().detach().requires_grad_(),
                    constraint=constraints.simplex,
                )
            )
        qs.append(pyro.param("qy", torch.tensor([0.75, 0.25], requires_grad=True)))
        qs = [q.unconstrained() for q in qs]
        data = (torch.rand(4, 3) > 0.5).to(dtype=qs[-1].dtype, device=qs[-1].device)

    with pyro_backend("pyro"):
        elbo = infer.TraceTMC_ELBO(max_plate_nesting=max_plate_nesting)
        enum_model = infer.config_enumerate(
            model,
            default="parallel",
            expand=False,
            num_samples=num_samples,
            tmc=tmc_strategy,
        )
        expected_loss = (-elbo.differentiable_loss(enum_model, lambda: None)).exp()
        expected_grads = grad(expected_loss, qs)

    with pyro_backend("contrib.funsor"):
        tmc = infer.TraceTMC_ELBO(max_plate_nesting=max_plate_nesting)
        tmc_model = infer.config_enumerate(
            model,
            default="parallel",
            expand=False,
            num_samples=num_samples,
            tmc=tmc_strategy,
        )
        actual_loss = (-tmc.differentiable_loss(tmc_model, lambda: None)).exp()
        actual_grads = grad(actual_loss, qs)

    prec = 0.05
    assert_equal(
        actual_loss,
        expected_loss,
        prec=prec,
        msg="".join(
            [
                "\nexpected loss = {}".format(expected_loss),
                "\n  actual loss = {}".format(actual_loss),
            ]
        ),
    )

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_equal(
            actual_grad,
            expected_grad,
            prec=prec,
            msg="".join(
                [
                    "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
                    "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
                ]
            ),
        )


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("num_samples,expand", [(400, False)])
@pytest.mark.parametrize("max_plate_nesting", [1])
@pytest.mark.parametrize("guide_type", ["prior", "factorized", "nonfactorized"])
@pytest.mark.parametrize("reparameterized", [False, True], ids=["dice", "pathwise"])
@pytest.mark.parametrize("tmc_strategy", ["diagonal", "mixture"])
def test_tmc_normals_chain_gradient(
    depth,
    num_samples,
    max_plate_nesting,
    expand,
    guide_type,
    reparameterized,
    tmc_strategy,
):
    def model(reparameterized):
        Normal = (
            dist.Normal
            if reparameterized
            else dist.testing.fakes.NonreparameterizedNormal
        )
        x = pyro.sample("x0", Normal(pyro.param("q2"), math.sqrt(1.0 / depth)))
        for i in range(1, depth):
            x = pyro.sample("x{}".format(i), Normal(x, math.sqrt(1.0 / depth)))
        pyro.sample("y", Normal(x, 1.0), obs=torch.tensor(float(1)))

    def factorized_guide(reparameterized):
        Normal = (
            dist.Normal
            if reparameterized
            else dist.testing.fakes.NonreparameterizedNormal
        )
        pyro.sample("x0", Normal(pyro.param("q2"), math.sqrt(1.0 / depth)))
        for i in range(1, depth):
            pyro.sample("x{}".format(i), Normal(0.0, math.sqrt(float(i + 1) / depth)))

    def nonfactorized_guide(reparameterized):
        Normal = (
            dist.Normal
            if reparameterized
            else dist.testing.fakes.NonreparameterizedNormal
        )
        x = pyro.sample("x0", Normal(pyro.param("q2"), math.sqrt(1.0 / depth)))
        for i in range(1, depth):
            x = pyro.sample("x{}".format(i), Normal(x, math.sqrt(1.0 / depth)))

    with pyro_backend("contrib.funsor"):
        # compare reparameterized and nonreparameterized gradient estimates
        q2 = pyro.param("q2", torch.tensor(0.5, requires_grad=True))
        qs = (q2.unconstrained(),)

        tmc = infer.TraceTMC_ELBO(max_plate_nesting=max_plate_nesting)
        tmc_model = infer.config_enumerate(
            model,
            default="parallel",
            expand=expand,
            num_samples=num_samples,
            tmc=tmc_strategy,
        )
        guide = (
            factorized_guide
            if guide_type == "factorized"
            else (
                nonfactorized_guide
                if guide_type == "nonfactorized"
                else lambda *args: None
            )
        )
        tmc_guide = infer.config_enumerate(
            guide,
            default="parallel",
            expand=expand,
            num_samples=num_samples,
            tmc=tmc_strategy,
        )

        # convert to linear space for unbiasedness
        actual_loss = (
            -tmc.differentiable_loss(tmc_model, tmc_guide, reparameterized)
        ).exp()
        actual_grads = grad(actual_loss, qs)

    # gold values from Funsor
    expected_grads = (
        torch.tensor({1: 0.0999, 2: 0.0860, 3: 0.0802, 4: 0.0771}[depth]),
    )

    grad_prec = 0.05 if reparameterized else 0.1

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        print(actual_loss)
        assert_equal(
            actual_grad,
            expected_grad,
            prec=grad_prec,
            msg="".join(
                [
                    "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
                    "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
                ]
            ),
        )
