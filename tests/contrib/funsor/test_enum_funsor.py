# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import pyroapi
import pytest
import torch
from torch.autograd import grad
from torch.distributions import constraints

from pyro.ops.indexing import Vindex
from pyro.util import torch_isnan
from tests.common import assert_equal, xfail_param

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor

    import pyro.contrib.funsor

    funsor.set_backend("torch")
    from pyroapi import distributions as dist
    from pyroapi import handlers, infer, pyro
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)

_PYRO_BACKEND = os.environ.get("TEST_ENUM_PYRO_BACKEND", "contrib.funsor")


def _check_loss_and_grads(expected_loss, actual_loss):
    assert_equal(
        actual_loss,
        expected_loss,
        msg="Expected:\n{}\nActual:\n{}".format(
            expected_loss.detach().cpu().numpy(), actual_loss.detach().cpu().numpy()
        ),
    )

    if "TEST_ENUM_PYRO_BACKEND" in os.environ:  # only log if we manually set a backend
        logging.debug(
            "Expected:\n{}\nActual:\n{}".format(
                expected_loss.detach().cpu().numpy(), actual_loss.detach().cpu().numpy()
            )
        )

    names = pyro.get_param_store().keys()
    params = [pyro.param(name).unconstrained() for name in names]
    actual_grads = grad(actual_loss, params, allow_unused=True, retain_graph=True)
    expected_grads = grad(expected_loss, params, allow_unused=True, retain_graph=True)
    for name, actual_grad, expected_grad in zip(names, actual_grads, expected_grads):
        if actual_grad is None or expected_grad is None:
            continue
        assert not torch_isnan(actual_grad)
        assert not torch_isnan(expected_grad)
        assert_equal(
            actual_grad,
            expected_grad,
            msg="{}\nExpected:\n{}\nActual:\n{}".format(
                name,
                expected_grad.detach().cpu().numpy(),
                actual_grad.detach().cpu().numpy(),
            ),
        )


@pytest.mark.parametrize("inner_dim", [2])
@pytest.mark.parametrize("outer_dim", [2])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_plate_plate(outer_dim, inner_dim):
    pyro.get_param_store().clear()
    q = pyro.param("q", torch.tensor([0.75, 0.25], requires_grad=True))
    p = 0.2693204236205713  # for which kl(Categorical(q), Categorical(p)) = 0.5
    p = torch.tensor([p, 1 - p])

    def model():
        d = dist.Categorical(p)
        context1 = pyro.plate("outer", outer_dim, dim=-1)
        context2 = pyro.plate("inner", inner_dim, dim=-2)
        pyro.sample("w", d)
        with context1:
            pyro.sample("x", d)
        with context2:
            pyro.sample("y", d)
        with context1, context2:
            pyro.sample("z", d)

    def guide():
        d = dist.Categorical(pyro.param("q"))
        context1 = pyro.plate("outer", outer_dim, dim=-1)
        context2 = pyro.plate("inner", inner_dim, dim=-2)
        pyro.sample("w", d, infer={"enumerate": "parallel"})
        with context1:
            pyro.sample("x", d, infer={"enumerate": "parallel"})
        with context2:
            pyro.sample("y", d, infer={"enumerate": "parallel"})
        with context1, context2:
            pyro.sample("z", d, infer={"enumerate": "parallel"})

    kl_node = torch.distributions.kl.kl_divergence(
        torch.distributions.Categorical(q), torch.distributions.Categorical(p)
    )
    kl = (1 + outer_dim + inner_dim + outer_dim * inner_dim) * kl_node
    expected_loss = kl
    expected_grad = grad(kl, [q.unconstrained()])[0]

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
    actual_loss = elbo.differentiable_loss(model, guide)
    actual_grad = grad(actual_loss, [q.unconstrained()])[0]

    assert_equal(actual_loss, expected_loss, prec=1e-5)
    assert_equal(actual_grad, expected_grad, prec=1e-5)


@pytest.mark.parametrize("scale", [1, 10])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_1(scale):
    pyro.param(
        "guide_probs_x", torch.tensor([0.1, 0.9]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_x", torch.tensor([0.4, 0.6]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_y",
        torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_z", torch.tensor([0.3, 0.7]), constraint=constraints.simplex
    )

    @handlers.scale(scale=scale)
    def auto_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        pyro.sample("y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"})
        pyro.sample("z", dist.Categorical(probs_z), obs=torch.tensor(0))

    @handlers.scale(scale=scale)
    def hand_model():
        probs_x = pyro.param("model_probs_x")
        probs_z = pyro.param("model_probs_z")
        pyro.sample("x", dist.Categorical(probs_x))
        pyro.sample("z", dist.Categorical(probs_z), obs=torch.tensor(0))

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def guide():
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_2(scale):
    pyro.param(
        "guide_probs_x", torch.tensor([0.1, 0.9]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_x", torch.tensor([0.4, 0.6]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_y",
        torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_z",
        torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
        constraint=constraints.simplex,
    )

    @handlers.scale(scale=scale)
    def auto_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        y = pyro.sample(
            "y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"}
        )
        pyro.sample("z", dist.Categorical(probs_z[y]), obs=torch.tensor(0))

    @handlers.scale(scale=scale)
    def hand_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        probs_yz = probs_y.mm(probs_z)
        x = pyro.sample("x", dist.Categorical(probs_x))
        pyro.sample("z", dist.Categorical(probs_yz[x]), obs=torch.tensor(0))

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def guide():
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_3(scale):
    pyro.param(
        "guide_probs_x", torch.tensor([0.1, 0.9]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_x", torch.tensor([0.4, 0.6]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_y",
        torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_z",
        torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
        constraint=constraints.simplex,
    )

    def auto_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            y = pyro.sample(
                "y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"}
            )
            pyro.sample("z", dist.Categorical(probs_z[y]), obs=torch.tensor(0))

    def hand_model():
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        probs_yz = probs_y.mm(probs_z)
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            pyro.sample("z", dist.Categorical(probs_yz[x]), obs=torch.tensor(0))

    @infer.config_enumerate
    def guide():
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pytest.mark.parametrize(
    "num_samples,num_masked", [(2, 2), (3, 2)], ids=["batch", "masked"]
)
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_1(num_samples, num_masked, scale):
    #              +---------+
    #  x ----> y ----> z     |
    #              |       N |
    #              +---------+
    pyro.param(
        "guide_probs_x", torch.tensor([0.1, 0.9]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_x", torch.tensor([0.4, 0.6]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_y",
        torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_z",
        torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
        constraint=constraints.simplex,
    )

    def auto_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            y = pyro.sample(
                "y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"}
            )
            if num_masked == num_samples:
                with pyro.plate("data", len(data)):
                    pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)
            else:
                with pyro.plate("data", len(data)):
                    with handlers.mask(mask=torch.arange(num_samples) < num_masked):
                        pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)

    def hand_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            y = pyro.sample(
                "y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"}
            )
            for i in pyro.plate("data", num_masked):
                pyro.sample("z_{}".format(i), dist.Categorical(probs_z[y]), obs=data[i])

    @infer.config_enumerate
    def guide(data):
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    data = dist.Categorical(torch.tensor([0.3, 0.7])).sample((num_samples,))
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, guide, data)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    hand_loss = elbo.differentiable_loss(hand_model, guide, data)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pytest.mark.parametrize(
    "num_samples,num_masked", [(2, 2), (3, 2)], ids=["batch", "masked"]
)
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_2(num_samples, num_masked, scale):
    #      +-----------------+
    #  x ----> y ----> z     |
    #      |               N |
    #      +-----------------+
    pyro.param(
        "guide_probs_x", torch.tensor([0.1, 0.9]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_x", torch.tensor([0.4, 0.6]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_y",
        torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_z",
        torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
        constraint=constraints.simplex,
    )

    def auto_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            with pyro.plate("data", len(data)):
                if num_masked == num_samples:
                    y = pyro.sample(
                        "y",
                        dist.Categorical(probs_y[x]),
                        infer={"enumerate": "parallel"},
                    )
                    pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)
                else:
                    with handlers.mask(mask=torch.arange(num_samples) < num_masked):
                        y = pyro.sample(
                            "y",
                            dist.Categorical(probs_y[x]),
                            infer={"enumerate": "parallel"},
                        )
                        pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)

    def hand_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        x = pyro.sample("x", dist.Categorical(probs_x))
        with handlers.scale(scale=scale):
            for i in pyro.plate("data", num_masked):
                y = pyro.sample(
                    "y_{}".format(i),
                    dist.Categorical(probs_y[x]),
                    infer={"enumerate": "parallel"},
                )
                pyro.sample("z_{}".format(i), dist.Categorical(probs_z[y]), obs=data[i])

    @infer.config_enumerate
    def guide(data):
        probs_x = pyro.param("guide_probs_x")
        pyro.sample("x", dist.Categorical(probs_x))

    data = dist.Categorical(torch.tensor([0.3, 0.7])).sample((num_samples,))
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, guide, data)
    hand_loss = elbo.differentiable_loss(hand_model, guide, data)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pytest.mark.parametrize(
    "num_samples,num_masked", [(2, 2), (3, 2)], ids=["batch", "masked"]
)
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_3(num_samples, num_masked, scale):
    #  +-----------------------+
    #  | x ----> y ----> z     |
    #  |                     N |
    #  +-----------------------+
    # This plate should remain unreduced since all enumeration is in a single plate.
    pyro.param(
        "guide_probs_x", torch.tensor([0.1, 0.9]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_x", torch.tensor([0.4, 0.6]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_y",
        torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_z",
        torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
        constraint=constraints.simplex,
    )

    @handlers.scale(scale=scale)
    def auto_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        with pyro.plate("data", len(data)):
            if num_masked == num_samples:
                x = pyro.sample("x", dist.Categorical(probs_x))
                y = pyro.sample(
                    "y", dist.Categorical(probs_y[x]), infer={"enumerate": "parallel"}
                )
                pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)
            else:
                with handlers.mask(mask=torch.arange(num_samples) < num_masked):
                    x = pyro.sample("x", dist.Categorical(probs_x))
                    y = pyro.sample(
                        "y",
                        dist.Categorical(probs_y[x]),
                        infer={"enumerate": "parallel"},
                    )
                    pyro.sample("z", dist.Categorical(probs_z[y]), obs=data)

    @handlers.scale(scale=scale)
    @infer.config_enumerate
    def auto_guide(data):
        probs_x = pyro.param("guide_probs_x")
        with pyro.plate("data", len(data)):
            if num_masked == num_samples:
                pyro.sample("x", dist.Categorical(probs_x))
            else:
                with handlers.mask(mask=torch.arange(num_samples) < num_masked):
                    pyro.sample("x", dist.Categorical(probs_x))

    @handlers.scale(scale=scale)
    def hand_model(data):
        probs_x = pyro.param("model_probs_x")
        probs_y = pyro.param("model_probs_y")
        probs_z = pyro.param("model_probs_z")
        for i in pyro.plate("data", num_masked):
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs_x))
            y = pyro.sample(
                "y_{}".format(i),
                dist.Categorical(probs_y[x]),
                infer={"enumerate": "parallel"},
            )
            pyro.sample("z_{}".format(i), dist.Categorical(probs_z[y]), obs=data[i])

    @handlers.scale(scale=scale)
    @infer.config_enumerate
    def hand_guide(data):
        probs_x = pyro.param("guide_probs_x")
        for i in pyro.plate("data", num_masked):
            pyro.sample("x_{}".format(i), dist.Categorical(probs_x))

    data = dist.Categorical(torch.tensor([0.3, 0.7])).sample((num_samples,))
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1, strict_enumeration_warning=False)
    auto_loss = elbo.differentiable_loss(auto_model, auto_guide, data)
    hand_loss = elbo.differentiable_loss(hand_model, hand_guide, data)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pytest.mark.parametrize(
    "outer_obs,inner_obs", [(False, True), (True, False), (True, True)]
)
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_4(outer_obs, inner_obs, scale):
    #    a ---> outer_obs
    #      \
    #  +-----\------------------+
    #  |       \                |
    #  | b ---> inner_obs   N=2 |
    #  +------------------------+
    # This tests two different observations, one outside and one inside an plate.
    pyro.param("probs_a", torch.tensor([0.4, 0.6]), constraint=constraints.simplex)
    pyro.param("probs_b", torch.tensor([0.6, 0.4]), constraint=constraints.simplex)
    pyro.param("locs", torch.tensor([-1.0, 1.0]))
    pyro.param("scales", torch.tensor([1.0, 2.0]), constraint=constraints.positive)
    outer_data = torch.tensor(2.0)
    inner_data = torch.tensor([0.5, 1.5])

    @handlers.scale(scale=scale)
    def auto_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        locs = pyro.param("locs")
        scales = pyro.param("scales")
        a = pyro.sample("a", dist.Categorical(probs_a), infer={"enumerate": "parallel"})
        if outer_obs:
            pyro.sample("outer_obs", dist.Normal(0.0, scales[a]), obs=outer_data)
        with pyro.plate("inner", 2):
            b = pyro.sample(
                "b", dist.Categorical(probs_b), infer={"enumerate": "parallel"}
            )
            if inner_obs:
                pyro.sample(
                    "inner_obs", dist.Normal(locs[b], scales[a]), obs=inner_data
                )

    @handlers.scale(scale=scale)
    def hand_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        locs = pyro.param("locs")
        scales = pyro.param("scales")
        a = pyro.sample("a", dist.Categorical(probs_a), infer={"enumerate": "parallel"})
        if outer_obs:
            pyro.sample("outer_obs", dist.Normal(0.0, scales[a]), obs=outer_data)
        for i in pyro.plate("inner", 2):
            b = pyro.sample(
                "b_{}".format(i),
                dist.Categorical(probs_b),
                infer={"enumerate": "parallel"},
            )
            if inner_obs:
                pyro.sample(
                    "inner_obs_{}".format(i),
                    dist.Normal(locs[b], scales[a]),
                    obs=inner_data[i],
                )

    def guide():
        pass

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.xfail(reason="Not supported in regular Pyro")
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_5():
    #        Guide   Model
    #                  a
    #  +---------------|--+
    #  | M=2           V  |
    #  |       b ----> c  |
    #  +------------------+
    pyro.param(
        "model_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_b", torch.tensor([0.6, 0.4]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_c",
        torch.tensor(
            [[[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]], [[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]]
        ),
        constraint=constraints.simplex,
    )
    pyro.param(
        "guide_probs_b", torch.tensor([0.8, 0.2]), constraint=constraints.simplex
    )
    data = torch.tensor([1, 2])

    @infer.config_enumerate
    def model_plate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", 2):
            b = pyro.sample("b", dist.Categorical(probs_b))
            pyro.sample("c", dist.Categorical(Vindex(probs_c)[a, b]), obs=data)

    @infer.config_enumerate
    def guide_plate():
        probs_b = pyro.param("guide_probs_b")
        with pyro.plate("b_axis", 2):
            pyro.sample("b", dist.Categorical(probs_b))

    @infer.config_enumerate
    def model_iplate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("b_axis", 2):
            b = pyro.sample("b_{}".format(i), dist.Categorical(probs_b))
            pyro.sample(
                "c_{}".format(i), dist.Categorical(Vindex(probs_c)[a, b]), obs=data[i]
            )

    @infer.config_enumerate
    def guide_iplate():
        probs_b = pyro.param("guide_probs_b")
        for i in pyro.plate("b_axis", 2):
            pyro.sample("b_{}".format(i), dist.Categorical(probs_b))

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    expected_loss = elbo.differentiable_loss(model_iplate, guide_iplate)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    with pytest.raises(
        ValueError, match="Expected model enumeration to be no more global than guide"
    ):
        actual_loss = elbo.differentiable_loss(model_plate, guide_plate)
        # This never gets run because we don't support this yet.
        _check_loss_and_grads(expected_loss, actual_loss)


@pytest.mark.parametrize("enumerate1", ["parallel", "sequential"])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_6(enumerate1):
    #     Guide           Model
    #           +-------+
    #       b ----> c <---- a
    #           |  M=2  |
    #           +-------+
    # This tests that sequential enumeration over b works, even though
    # model-side enumeration moves c into b's plate via contraction.
    pyro.param(
        "model_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_b", torch.tensor([0.6, 0.4]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_c",
        torch.tensor(
            [[[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]], [[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]]
        ),
        constraint=constraints.simplex,
    )
    pyro.param(
        "guide_probs_b", torch.tensor([0.8, 0.2]), constraint=constraints.simplex
    )
    data = torch.tensor([1, 2])

    @infer.config_enumerate
    def model_plate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample("b", dist.Categorical(probs_b))
        with pyro.plate("b_axis", 2):
            pyro.sample("c", dist.Categorical(Vindex(probs_c)[a, b]), obs=data)

    @infer.config_enumerate
    def model_iplate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample("b", dist.Categorical(probs_b))
        for i in pyro.plate("b_axis", 2):
            pyro.sample(
                "c_{}".format(i), dist.Categorical(Vindex(probs_c)[a, b]), obs=data[i]
            )

    @infer.config_enumerate(default=enumerate1)
    def guide():
        probs_b = pyro.param("guide_probs_b")
        pyro.sample("b", dist.Categorical(probs_b))

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    expected_loss = elbo.differentiable_loss(model_iplate, guide)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    actual_loss = elbo.differentiable_loss(model_plate, guide)
    _check_loss_and_grads(expected_loss, actual_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_7(scale):
    #  Guide    Model
    #    a -----> b
    #    |        |
    #  +-|--------|----------------+
    #  | V        V                |
    #  | c -----> d -----> e   N=2 |
    #  +---------------------------+
    # This tests a mixture of model and guide enumeration.
    pyro.param(
        "model_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_b",
        torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_c",
        torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_d",
        torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_e",
        torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "guide_probs_a", torch.tensor([0.35, 0.64]), constraint=constraints.simplex
    )
    pyro.param(
        "guide_probs_c",
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),  # deterministic
        constraint=constraints.simplex,
    )

    @handlers.scale(scale=scale)
    def auto_model(data):
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        probs_d = pyro.param("model_probs_d")
        probs_e = pyro.param("model_probs_e")
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample(
            "b", dist.Categorical(probs_b[a]), infer={"enumerate": "parallel"}
        )
        with pyro.plate("data", 2):
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
            d = pyro.sample(
                "d",
                dist.Categorical(Vindex(probs_d)[b, c]),
                infer={"enumerate": "parallel"},
            )
            pyro.sample("obs", dist.Categorical(probs_e[d]), obs=data)

    @handlers.scale(scale=scale)
    def auto_guide(data):
        probs_a = pyro.param("guide_probs_a")
        probs_c = pyro.param("guide_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a), infer={"enumerate": "parallel"})
        with pyro.plate("data", 2):
            pyro.sample("c", dist.Categorical(probs_c[a]))

    @handlers.scale(scale=scale)
    def hand_model(data):
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        probs_d = pyro.param("model_probs_d")
        probs_e = pyro.param("model_probs_e")
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = pyro.sample(
            "b", dist.Categorical(probs_b[a]), infer={"enumerate": "parallel"}
        )
        for i in pyro.plate("data", 2):
            c = pyro.sample("c_{}".format(i), dist.Categorical(probs_c[a]))
            d = pyro.sample(
                "d_{}".format(i),
                dist.Categorical(Vindex(probs_d)[b, c]),
                infer={"enumerate": "parallel"},
            )
            pyro.sample("obs_{}".format(i), dist.Categorical(probs_e[d]), obs=data[i])

    @handlers.scale(scale=scale)
    def hand_guide(data):
        probs_a = pyro.param("guide_probs_a")
        probs_c = pyro.param("guide_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a), infer={"enumerate": "parallel"})
        for i in pyro.plate("data", 2):
            pyro.sample("c_{}".format(i), dist.Categorical(probs_c[a]))

    data = torch.tensor([0, 0])
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, auto_guide, data)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, hand_guide, data)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plates_1(scale):
    #  +-----------------+
    #  | a ----> b   M=2 |
    #  +-----------------+
    #  +-----------------+
    #  | c ----> d   N=3 |
    #  +-----------------+
    # This tests two unrelated plates.
    # Each should remain uncontracted.
    pyro.param("probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex)
    pyro.param(
        "probs_b",
        torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
        constraint=constraints.simplex,
    )
    pyro.param("probs_c", torch.tensor([0.75, 0.25]), constraint=constraints.simplex)
    pyro.param(
        "probs_d",
        torch.tensor([[0.4, 0.6], [0.3, 0.7]]),
        constraint=constraints.simplex,
    )
    b_data = torch.tensor([0, 1])
    d_data = torch.tensor([0, 0, 1])

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def auto_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        with pyro.plate("a_axis", 2):
            a = pyro.sample("a", dist.Categorical(probs_a))
            pyro.sample("b", dist.Categorical(probs_b[a]), obs=b_data)
        with pyro.plate("c_axis", 3):
            c = pyro.sample("c", dist.Categorical(probs_c))
            pyro.sample("d", dist.Categorical(probs_d[c]), obs=d_data)

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def hand_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        for i in pyro.plate("a_axis", 2):
            a = pyro.sample("a_{}".format(i), dist.Categorical(probs_a))
            pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]), obs=b_data[i])
        for j in pyro.plate("c_axis", 3):
            c = pyro.sample("c_{}".format(j), dist.Categorical(probs_c))
            pyro.sample("d_{}".format(j), dist.Categorical(probs_d[c]), obs=d_data[j])

    def guide():
        pass

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plates_2(scale):
    #  +---------+       +---------+
    #  |     b <---- a ----> c     |
    #  | M=2     |       |     N=3 |
    #  +---------+       +---------+
    # This tests two different plates with recycled dimension.
    pyro.param("probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex)
    pyro.param(
        "probs_b",
        torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "probs_c",
        torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
        constraint=constraints.simplex,
    )
    b_data = torch.tensor([0, 1])
    c_data = torch.tensor([0, 0, 1])

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def auto_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", 2):
            pyro.sample("b", dist.Categorical(probs_b[a]), obs=b_data)
        with pyro.plate("c_axis", 3):
            pyro.sample("c", dist.Categorical(probs_c[a]), obs=c_data)

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def hand_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("b_axis", 2):
            pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]), obs=b_data[i])
        for j in pyro.plate("c_axis", 3):
            pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a]), obs=c_data[j])

    def guide():
        pass

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plates_3(scale):
    #      +--------------------+
    #      |  +----------+      |
    #  a -------> b      |      |
    #      |  |      N=2 |      |
    #      |  +----------+  M=2 |
    #      +--------------------+
    # This is tests the case of multiple plate contractions in
    # a single step.
    pyro.param("probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex)
    pyro.param(
        "probs_b",
        torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
        constraint=constraints.simplex,
    )
    data = torch.tensor([[0, 1], [0, 0]])

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def auto_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("outer", 2):
            with pyro.plate("inner", 2):
                pyro.sample("b", dist.Categorical(probs_b[a]), obs=data)

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def hand_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        inner = pyro.plate("inner", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("outer", 2):
            for j in inner:
                pyro.sample(
                    "b_{}_{}".format(i, j), dist.Categorical(probs_b[a]), obs=data[i, j]
                )

    def guide():
        pass

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plates_4(scale):
    #      +--------------------+
    #      |       +----------+ |
    #  a ----> b ----> c      | |
    #      |       |      N=2 | |
    #      | M=2   +----------+ |
    #      +--------------------+
    pyro.param("probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex)
    pyro.param(
        "probs_b",
        torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "probs_c",
        torch.tensor([[0.4, 0.6], [0.3, 0.7]]),
        constraint=constraints.simplex,
    )

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def auto_model(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("inner", 2):
                pyro.sample("c", dist.Categorical(probs_c[b]), obs=data)

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def hand_model(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        inner = pyro.plate("inner", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("outer", 2):
            b = pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]))
            for j in inner:
                pyro.sample(
                    "c_{}_{}".format(i, j), dist.Categorical(probs_c[b]), obs=data[i, j]
                )

    def guide(data):
        pass

    data = torch.tensor([[0, 1], [0, 0]])
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
    auto_loss = elbo.differentiable_loss(auto_model, guide, data)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide, data)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plates_5(scale):
    #     a
    #     | \
    #  +--|---\------------+
    #  |  V   +-\--------+ |
    #  |  b ----> c      | |
    #  |      |      N=2 | |
    #  | M=2  +----------+ |
    #  +-------------------+
    pyro.param("probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex)
    pyro.param(
        "probs_b",
        torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "probs_c",
        torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.2, 0.8], [0.1, 0.9]]]),
        constraint=constraints.simplex,
    )
    data = torch.tensor([[0, 1], [0, 0]])

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def auto_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("inner", 2):
                pyro.sample("c", dist.Categorical(Vindex(probs_c)[a, b]), obs=data)

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def hand_model():
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        inner = pyro.plate("inner", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("outer", 2):
            b = pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]))
            for j in inner:
                pyro.sample(
                    "c_{}_{}".format(i, j),
                    dist.Categorical(Vindex(probs_c)[a, b]),
                    obs=data[i, j],
                )

    def guide():
        pass

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
    auto_loss = elbo.differentiable_loss(auto_model, guide)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    hand_loss = elbo.differentiable_loss(hand_model, guide)
    _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize("scale", [1, 10])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plates_6(scale):
    #         +----------+
    #         |      M=2 |
    #     a ----> b      |
    #     |   |   |      |
    #  +--|-------|--+   |
    #  |  V   |   V  |   |
    #  |  c ----> d  |   |
    #  |      |      |   |
    #  | N=2  +------|---+
    #  +-------------+
    # This tests different ways of mixing two independence contexts,
    # where each can be either sequential or vectorized plate.
    pyro.param("probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex)
    pyro.param(
        "probs_b",
        torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "probs_c",
        torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "probs_d",
        torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]]),
        constraint=constraints.simplex,
    )

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    @handlers.trace
    def model_iplate_iplate(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        b_axis = pyro.plate("b_axis", 2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = [
            pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a])) for i in b_axis
        ]
        c = [
            pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a])) for j in c_axis
        ]
        for i in b_axis:
            for j in c_axis:
                pyro.sample(
                    "d_{}_{}".format(i, j),
                    dist.Categorical(Vindex(probs_d)[b[i], c[j]]),
                    obs=data[i, j],
                )

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def model_iplate_plate(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        b_axis = pyro.plate("b_axis", 2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        for i in b_axis:
            b_i = pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]))
            with c_axis:
                pyro.sample(
                    "d_{}".format(i),
                    dist.Categorical(Vindex(probs_d)[b_i, c]),
                    obs=data[i],
                )

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    @handlers.trace
    def model_plate_iplate(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        b_axis = pyro.plate("b_axis", 2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        c = [
            pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a])) for j in c_axis
        ]
        with b_axis:
            for j in c_axis:
                pyro.sample(
                    "d_{}".format(j),
                    dist.Categorical(Vindex(probs_d)[b, c[j]]),
                    obs=data[:, j],
                )

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def model_plate_plate(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        b_axis = pyro.plate("b_axis", 2, dim=-1)
        c_axis = pyro.plate("c_axis", 2, dim=-2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        with b_axis, c_axis:
            pyro.sample("d", dist.Categorical(Vindex(probs_d)[b, c]), obs=data)

    def guide(data):
        pass

    # Check that either one of the sequential plates can be promoted to be vectorized.
    data = torch.tensor([[0, 1], [0, 0]])
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    loss_iplate_iplate = elbo.differentiable_loss(model_iplate_iplate, guide, data)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    loss_plate_iplate = elbo.differentiable_loss(model_plate_iplate, guide, data)
    loss_iplate_plate = elbo.differentiable_loss(model_iplate_plate, guide, data)
    _check_loss_and_grads(loss_iplate_iplate, loss_iplate_plate)
    _check_loss_and_grads(loss_iplate_iplate, loss_plate_iplate)

    # But promoting both to plates should result in an error.
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
    with pytest.raises(ValueError, match="intractable!"):
        elbo.differentiable_loss(model_plate_plate, guide, data)


@pytest.mark.parametrize("scale", [1, 10])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plates_7(scale):
    #         +-------------+
    #         |         N=2 |
    #     a -------> c      |
    #     |   |      |      |
    #  +--|----------|--+   |
    #  |  |   |      V  |   |
    #  |  V   |      e  |   |
    #  |  b ----> d     |   |
    #  |      |         |   |
    #  | M=2  +---------|---+
    #  +----------------+
    # This tests tree-structured dependencies among variables but
    # non-tree dependencies among plate nestings.
    pyro.param("probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex)
    pyro.param(
        "probs_b",
        torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "probs_c",
        torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "probs_d",
        torch.tensor([[0.3, 0.7], [0.2, 0.8]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "probs_e",
        torch.tensor([[0.4, 0.6], [0.3, 0.7]]),
        constraint=constraints.simplex,
    )

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    @handlers.trace
    def model_iplate_iplate(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        probs_e = pyro.param("probs_e")
        b_axis = pyro.plate("b_axis", 2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        b = [
            pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a])) for i in b_axis
        ]
        c = [
            pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a])) for j in c_axis
        ]
        for i in b_axis:
            for j in c_axis:
                pyro.sample(
                    "d_{}_{}".format(i, j),
                    dist.Categorical(probs_d[b[i]]),
                    obs=data[i, j],
                )
                pyro.sample(
                    "e_{}_{}".format(i, j),
                    dist.Categorical(probs_e[c[j]]),
                    obs=data[i, j],
                )

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def model_iplate_plate(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        probs_e = pyro.param("probs_e")
        b_axis = pyro.plate("b_axis", 2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        for i in b_axis:
            b_i = pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]))
            with c_axis:
                pyro.sample(
                    "d_{}".format(i), dist.Categorical(probs_d[b_i]), obs=data[i]
                )
                pyro.sample("e_{}".format(i), dist.Categorical(probs_e[c]), obs=data[i])

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    @handlers.trace
    def model_plate_iplate(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        probs_e = pyro.param("probs_e")
        b_axis = pyro.plate("b_axis", 2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        c = [
            pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a])) for j in c_axis
        ]
        with b_axis:
            for j in c_axis:
                pyro.sample(
                    "d_{}".format(j), dist.Categorical(probs_d[b]), obs=data[:, j]
                )
                pyro.sample(
                    "e_{}".format(j), dist.Categorical(probs_e[c[j]]), obs=data[:, j]
                )

    @infer.config_enumerate
    @handlers.scale(scale=scale)
    def model_plate_plate(data):
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        probs_e = pyro.param("probs_e")
        b_axis = pyro.plate("b_axis", 2, dim=-1)
        c_axis = pyro.plate("c_axis", 2, dim=-2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        with b_axis, c_axis:
            pyro.sample("d", dist.Categorical(probs_d[b]), obs=data)
            pyro.sample("e", dist.Categorical(probs_e[c]), obs=data)

    def guide(data):
        pass

    # Check that any combination of sequential plates can be promoted to be vectorized.
    data = torch.tensor([[0, 1], [0, 0]])
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    loss_iplate_iplate = elbo.differentiable_loss(model_iplate_iplate, guide, data)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    loss_plate_iplate = elbo.differentiable_loss(model_plate_iplate, guide, data)
    loss_iplate_plate = elbo.differentiable_loss(model_iplate_plate, guide, data)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
    loss_plate_plate = elbo.differentiable_loss(model_plate_plate, guide, data)
    _check_loss_and_grads(loss_iplate_iplate, loss_plate_iplate)
    _check_loss_and_grads(loss_iplate_iplate, loss_iplate_plate)
    _check_loss_and_grads(loss_iplate_iplate, loss_plate_plate)


@pytest.mark.parametrize("guide_scale", [1])
@pytest.mark.parametrize("model_scale", [1])
@pytest.mark.parametrize(
    "outer_vectorized",
    [False, xfail_param(True, reason="validation not yet implemented")],
)
@pytest.mark.parametrize("inner_vectorized", [False, True])
@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plates_8(
    model_scale, guide_scale, inner_vectorized, outer_vectorized
):
    #        Guide   Model
    #                  a
    #      +-----------|--------+
    #      | M=2   +---|------+ |
    #      |       |   V  N=2 | |
    #      |   b ----> c      | |
    #      |       +----------+ |
    #      +--------------------+
    pyro.param(
        "model_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_b", torch.tensor([0.6, 0.4]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_c",
        torch.tensor(
            [[[0.4, 0.5, 0.1], [0.3, 0.5, 0.2]], [[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]]
        ),
        constraint=constraints.simplex,
    )
    pyro.param(
        "guide_probs_b", torch.tensor([0.8, 0.2]), constraint=constraints.simplex
    )
    data = torch.tensor([[0, 1], [0, 2]])

    @infer.config_enumerate
    @handlers.scale(scale=model_scale)
    def model_plate_plate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b))
            with pyro.plate("inner", 2):
                pyro.sample("c", dist.Categorical(Vindex(probs_c)[a, b]), obs=data)

    @infer.config_enumerate
    @handlers.scale(scale=model_scale)
    def model_iplate_plate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        inner = pyro.plate("inner", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("outer", 2):
            b = pyro.sample("b_{}".format(i), dist.Categorical(probs_b))
            with inner:
                pyro.sample(
                    "c_{}".format(i),
                    dist.Categorical(Vindex(probs_c)[a, b]),
                    obs=data[:, i],
                )

    @infer.config_enumerate
    @handlers.scale(scale=model_scale)
    def model_plate_iplate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("outer", 2):
            b = pyro.sample("b", dist.Categorical(probs_b))
            for j in pyro.plate("inner", 2):
                pyro.sample(
                    "c_{}".format(j),
                    dist.Categorical(Vindex(probs_c)[a, b]),
                    obs=data[j],
                )

    @infer.config_enumerate
    @handlers.scale(scale=model_scale)
    def model_iplate_iplate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        inner = pyro.plate("inner", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("outer", 2):
            b = pyro.sample("b_{}".format(i), dist.Categorical(probs_b))
            for j in inner:
                pyro.sample(
                    "c_{}_{}".format(i, j),
                    dist.Categorical(Vindex(probs_c)[a, b]),
                    obs=data[j, i],
                )

    @infer.config_enumerate
    @handlers.scale(scale=guide_scale)
    def guide_plate():
        probs_b = pyro.param("guide_probs_b")
        with pyro.plate("outer", 2):
            pyro.sample("b", dist.Categorical(probs_b))

    @infer.config_enumerate
    @handlers.scale(scale=guide_scale)
    def guide_iplate():
        probs_b = pyro.param("guide_probs_b")
        for i in pyro.plate("outer", 2):
            pyro.sample("b_{}".format(i), dist.Categorical(probs_b))

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    expected_loss = elbo.differentiable_loss(model_iplate_iplate, guide_iplate)
    if inner_vectorized:
        if outer_vectorized:
            elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
            actual_loss = elbo.differentiable_loss(model_plate_plate, guide_plate)
        else:
            elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
            actual_loss = elbo.differentiable_loss(model_iplate_plate, guide_iplate)
    else:
        if outer_vectorized:
            elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
            actual_loss = elbo.differentiable_loss(model_plate_iplate, guide_plate)
        else:
            elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
            actual_loss = elbo.differentiable_loss(model_iplate_iplate, guide_iplate)
    _check_loss_and_grads(expected_loss, actual_loss)


@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_9():
    #        Model   Guide
    #          a
    #  +-------|-------+
    #  | M=2   V       |
    #  |       b -> c  |
    #  +---------------+
    pyro.param(
        "model_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_b",
        torch.tensor([[0.3, 0.7], [0.6, 0.4]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_c",
        torch.tensor([[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "guide_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "guide_probs_b",
        torch.tensor([[0.3, 0.7], [0.8, 0.2]]),
        constraint=constraints.simplex,
    )
    data = torch.tensor([1, 2])

    @infer.config_enumerate
    def model_plate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", 2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            pyro.sample("c", dist.Categorical(probs_c[b]), obs=data)

    @infer.config_enumerate
    def guide_plate():
        probs_a = pyro.param("guide_probs_a")
        probs_b = pyro.param("guide_probs_b")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("b_axis", 2):
            pyro.sample("b", dist.Categorical(probs_b[a]))

    @infer.config_enumerate
    def model_iplate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("b_axis", 2):
            b = pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            pyro.sample(f"c_{i}", dist.Categorical(probs_c[b]), obs=data[i])

    @infer.config_enumerate
    def guide_iplate():
        probs_a = pyro.param("guide_probs_a")
        probs_b = pyro.param("guide_probs_b")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("b_axis", 2):
            pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    expected_loss = elbo.differentiable_loss(model_iplate, guide_iplate)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
    actual_loss = elbo.differentiable_loss(model_plate, guide_plate)
    _check_loss_and_grads(expected_loss, actual_loss)


@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_10():
    # Model
    # a -> [ [ bij -> cij ] ]
    # Guide
    # a -> [ [ bij ] ]
    pyro.param(
        "model_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_b",
        torch.tensor([[0.3, 0.7], [0.6, 0.4]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_c",
        torch.tensor([[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "guide_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "guide_probs_b",
        torch.tensor([[0.3, 0.7], [0.8, 0.2]]),
        constraint=constraints.simplex,
    )
    data = torch.tensor([[0, 1, 2], [1, 2, 2]])

    @infer.config_enumerate
    def model_plate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            with pyro.plate("j", 3, dim=-1):
                b = pyro.sample("b", dist.Categorical(probs_b[a]))
                pyro.sample("c", dist.Categorical(probs_c[b]), obs=data)

    @infer.config_enumerate
    def guide_plate():
        probs_a = pyro.param("guide_probs_a")
        probs_b = pyro.param("guide_probs_b")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            with pyro.plate("j", 3, dim=-1):
                pyro.sample("b", dist.Categorical(probs_b[a]))

    @infer.config_enumerate
    def model_iplate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("i", 2):
            for j in pyro.plate("j", 3):
                b = pyro.sample(f"b_{i}_{j}", dist.Categorical(probs_b[a]))
                pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[b]), obs=data[i, j])

    @infer.config_enumerate
    def guide_iplate():
        probs_a = pyro.param("guide_probs_a")
        probs_b = pyro.param("guide_probs_b")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("i", 2):
            for j in pyro.plate("j", 3):
                pyro.sample(f"b_{i}_{j}", dist.Categorical(probs_b[a]))

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    expected_loss = elbo.differentiable_loss(model_iplate, guide_iplate)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
    actual_loss = elbo.differentiable_loss(model_plate, guide_plate)
    _check_loss_and_grads(expected_loss, actual_loss)


@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_11():
    # Model
    # [ ai -> [ bij -> cij ] ]
    # Guide
    # [ ai -> [ bij ] ]
    pyro.param(
        "model_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_b",
        torch.tensor([[0.3, 0.7], [0.6, 0.4]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_c",
        torch.tensor([[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "guide_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "guide_probs_b",
        torch.tensor([[0.3, 0.7], [0.8, 0.2]]),
        constraint=constraints.simplex,
    )
    data = torch.tensor([[0, 1, 2], [1, 2, 2]])

    @infer.config_enumerate
    def model_plate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        with pyro.plate("i", 2, dim=-2):
            a = pyro.sample("a", dist.Categorical(probs_a))
            with pyro.plate("j", 3, dim=-1):
                b = pyro.sample("b", dist.Categorical(probs_b[a]))
                pyro.sample("c", dist.Categorical(probs_c[b]), obs=data)

    @infer.config_enumerate
    def guide_plate():
        probs_a = pyro.param("guide_probs_a")
        probs_b = pyro.param("guide_probs_b")
        with pyro.plate("i", 2, dim=-2):
            a = pyro.sample("a", dist.Categorical(probs_a))
            with pyro.plate("j", 3, dim=-1):
                pyro.sample("b", dist.Categorical(probs_b[a]))

    @infer.config_enumerate
    def model_iplate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        for i in pyro.plate("i", 2):
            a = pyro.sample(f"a_{i}", dist.Categorical(probs_a))
            for j in pyro.plate("j", 3):
                b = pyro.sample(f"b_{i}_{j}", dist.Categorical(probs_b[a]))
                pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[b]), obs=data[i, j])

    @infer.config_enumerate
    def guide_iplate():
        probs_a = pyro.param("guide_probs_a")
        probs_b = pyro.param("guide_probs_b")
        for i in pyro.plate("i", 2):
            a = pyro.sample(f"a_{i}", dist.Categorical(probs_a))
            for j in pyro.plate("j", 3):
                pyro.sample(f"b_{i}_{j}", dist.Categorical(probs_b[a]))

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    expected_loss = elbo.differentiable_loss(model_iplate, guide_iplate)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
    actual_loss = elbo.differentiable_loss(model_plate, guide_plate)
    _check_loss_and_grads(expected_loss, actual_loss)


@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_12():
    # Model
    # a -> [ bi -> [ cij -> dij ] ]
    # Guide
    # a -> [ bi -> [ cij ] ]
    pyro.param(
        "model_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_b",
        torch.tensor([[0.3, 0.7], [0.6, 0.4]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_c",
        torch.tensor([[0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_d",
        torch.tensor([[0.1, 0.6, 0.3], [0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "guide_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "guide_probs_b",
        torch.tensor([[0.3, 0.7], [0.8, 0.2]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "guide_probs_c",
        torch.tensor([[0.3, 0.3, 0.4], [0.2, 0.4, 0.4]]),
        constraint=constraints.simplex,
    )
    data = torch.tensor([[0, 1, 2], [1, 2, 2]])

    @infer.config_enumerate
    def model_plate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        probs_d = pyro.param("model_probs_d")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("j", 3, dim=-1):
                c = pyro.sample("c", dist.Categorical(probs_c[b]))
                pyro.sample("d", dist.Categorical(probs_d[c]), obs=data)

    @infer.config_enumerate
    def guide_plate():
        probs_a = pyro.param("guide_probs_a")
        probs_b = pyro.param("guide_probs_b")
        probs_c = pyro.param("guide_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("j", 3, dim=-1):
                pyro.sample("c", dist.Categorical(probs_c[b]))

    @infer.config_enumerate
    def model_iplate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        probs_d = pyro.param("model_probs_d")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("i", 2):
            b = pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            for j in pyro.plate("j", 3):
                c = pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[b]))
                pyro.sample(f"d_{i}_{j}", dist.Categorical(probs_d[c]), obs=data[i, j])

    @infer.config_enumerate
    def guide_iplate():
        probs_a = pyro.param("guide_probs_a")
        probs_b = pyro.param("guide_probs_b")
        probs_c = pyro.param("guide_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("i", 2):
            b = pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            for j in pyro.plate("j", 3):
                pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[b]))

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    expected_loss = elbo.differentiable_loss(model_iplate, guide_iplate)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
    actual_loss = elbo.differentiable_loss(model_plate, guide_plate)
    _check_loss_and_grads(expected_loss, actual_loss)


@pyroapi.pyro_backend(_PYRO_BACKEND)
def test_elbo_enumerate_plate_13():
    # Model
    # a -> [ cj -> [ dij ] ]
    # |
    # v
    # [ bi ]
    # Guide
    # a -> [ cj ]
    # |
    # v
    # [ bi ]
    pyro.param(
        "model_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "model_probs_b",
        torch.tensor([[0.3, 0.7], [0.6, 0.4]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_c",
        torch.tensor([[0.3, 0.7], [0.4, 0.6]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "model_probs_d",
        torch.tensor([[0.1, 0.6, 0.3], [0.3, 0.4, 0.3], [0.4, 0.4, 0.2]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "guide_probs_a", torch.tensor([0.45, 0.55]), constraint=constraints.simplex
    )
    pyro.param(
        "guide_probs_b",
        torch.tensor([[0.3, 0.7], [0.8, 0.2]]),
        constraint=constraints.simplex,
    )
    pyro.param(
        "guide_probs_c",
        torch.tensor([[0.2, 0.8], [0.9, 0.1]]),
        constraint=constraints.simplex,
    )
    data = torch.tensor([[0, 1, 2], [1, 2, 2]])

    @infer.config_enumerate
    def model_plate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        probs_d = pyro.param("model_probs_d")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("j", 3, dim=-1):
                c = pyro.sample("c", dist.Categorical(probs_c[a]))
                pyro.sample("d", dist.Categorical(probs_d[c]), obs=data)

    @infer.config_enumerate
    def guide_plate():
        probs_a = pyro.param("guide_probs_a")
        probs_b = pyro.param("guide_probs_b")
        probs_c = pyro.param("guide_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        with pyro.plate("i", 2, dim=-2):
            pyro.sample("b", dist.Categorical(probs_b[a]))
            with pyro.plate("j", 3, dim=-1):
                pyro.sample("c", dist.Categorical(probs_c[a]))

    @infer.config_enumerate
    def model_iplate():
        probs_a = pyro.param("model_probs_a")
        probs_b = pyro.param("model_probs_b")
        probs_c = pyro.param("model_probs_c")
        probs_d = pyro.param("model_probs_d")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("i", 2):
            pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            for j in pyro.plate("j", 3):
                c = pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[a]))
                pyro.sample(f"d_{i}_{j}", dist.Categorical(probs_d[c]), obs=data[i, j])

    @infer.config_enumerate
    def guide_iplate():
        probs_a = pyro.param("guide_probs_a")
        probs_b = pyro.param("guide_probs_b")
        probs_c = pyro.param("guide_probs_c")
        a = pyro.sample("a", dist.Categorical(probs_a))
        for i in pyro.plate("i", 2):
            pyro.sample(f"b_{i}", dist.Categorical(probs_b[a]))
            for j in pyro.plate("j", 3):
                pyro.sample(f"c_{i}_{j}", dist.Categorical(probs_c[a]))

    elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
    expected_loss = elbo.differentiable_loss(model_iplate, guide_iplate)
    elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
    actual_loss = elbo.differentiable_loss(model_plate, guide_plate)
    _check_loss_and_grads(expected_loss, actual_loss)
