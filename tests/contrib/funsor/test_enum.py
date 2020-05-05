# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import math

import pytest
import torch
from torch.autograd import grad
from torch.distributions import constraints

import funsor
import pyro.contrib.funsor
from pyro.ops.indexing import Vindex

from pyroapi import infer, pyro, pyro_backend
from pyroapi import distributions as dist
from tests.common import assert_equal


funsor.set_backend("torch")
torch.set_default_dtype(torch.float32)

logger = logging.getLogger(__name__)


def _check_loss_and_grads(expected_loss, actual_loss):
    assert_equal(actual_loss, expected_loss,
                 msg='Expected:\n{}\nActual:\n{}'.format(expected_loss.detach().cpu().numpy(),
                                                         actual_loss.detach().cpu().numpy()))

    names = pyro.get_param_store().keys()
    params = [pyro.param(name).unconstrained() for name in names]
    actual_grads = grad(actual_loss, params, allow_unused=True, retain_graph=True)
    expected_grads = grad(expected_loss, params, allow_unused=True, retain_graph=True)
    for name, actual_grad, expected_grad in zip(names, actual_grads, expected_grads):
        if actual_grad is None or expected_grad is None:
            continue
        assert_equal(actual_grad, expected_grad,
                     msg='{}\nExpected:\n{}\nActual:\n{}'.format(name,
                                                                 expected_grad.detach().cpu().numpy(),
                                                                 actual_grad.detach().cpu().numpy()))


@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("num_samples", [None, 200])
@pytest.mark.parametrize("max_plate_nesting", [2, 3])
@pytest.mark.parametrize("tmc_strategy", ["diagonal", "mixture"])
def test_tmc_categoricals(depth, max_plate_nesting, num_samples, tmc_strategy):

    def model():
        x = pyro.sample("x0", dist.Categorical(pyro.param("q0")))
        with pyro.plate("local", 3):
            for i in range(1, depth):
                x = pyro.sample("x{}".format(i),
                                dist.Categorical(pyro.param("q{}".format(i))[..., x, :]))
            with pyro.plate("data", 4):
                pyro.sample("y", dist.Bernoulli(pyro.param("qy")[..., x]),
                            obs=data)

    with pyro_backend("pyro"):
        # initialize
        qs = [pyro.param("q0", torch.tensor([0.4, 0.6], requires_grad=True))]
        for i in range(1, depth):
            qs.append(pyro.param(
                "q{}".format(i),
                torch.randn(2, 2).abs().detach().requires_grad_(),
                constraint=constraints.simplex
            ))
        qs.append(pyro.param("qy", torch.tensor([0.75, 0.25], requires_grad=True)))
        qs = [q.unconstrained() for q in qs]
        data = (torch.rand(4, 3) > 0.5).to(dtype=qs[-1].dtype, device=qs[-1].device)

    with pyro_backend("pyro"):
        elbo = infer.TraceTMC_ELBO(max_plate_nesting=max_plate_nesting)
        enum_model = infer.config_enumerate(
            model, default="parallel", expand=False, num_samples=num_samples, tmc=tmc_strategy)
        expected_loss = (-elbo.differentiable_loss(enum_model, lambda: None)).exp()
        expected_grads = grad(expected_loss, qs)

    with pyro_backend("contrib.funsor"):
        tmc = infer.TraceTMC_ELBO(max_plate_nesting=max_plate_nesting)
        tmc_model = infer.config_enumerate(
            model, default="parallel", expand=False, num_samples=num_samples, tmc=tmc_strategy)
        actual_loss = (-tmc.differentiable_loss(tmc_model, lambda: None)).exp()
        actual_grads = grad(actual_loss, qs)

    prec = 0.05
    assert_equal(actual_loss, expected_loss, prec=prec, msg="".join([
        "\nexpected loss = {}".format(expected_loss),
        "\n  actual loss = {}".format(actual_loss),
    ]))

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert_equal(actual_grad, expected_grad, prec=prec, msg="".join([
            "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
            "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
        ]))


@pytest.mark.parametrize("depth", [1, 2, 3, 4])
@pytest.mark.parametrize("num_samples,expand", [(200, False)])
@pytest.mark.parametrize("max_plate_nesting", [1])
@pytest.mark.parametrize("guide_type", ["prior", "factorized", "nonfactorized"])
@pytest.mark.parametrize("reparameterized", [False, True], ids=["dice", "pathwise"])
@pytest.mark.parametrize("tmc_strategy", ["diagonal", "mixture"])
def test_tmc_normals_chain_gradient(depth, num_samples, max_plate_nesting, expand,
                                    guide_type, reparameterized, tmc_strategy):
    def model(reparameterized):
        Normal = dist.Normal if reparameterized else dist.testing.fakes.NonreparameterizedNormal
        x = pyro.sample("x0", Normal(pyro.param("q2"), math.sqrt(1. / depth)))
        for i in range(1, depth):
            x = pyro.sample("x{}".format(i), Normal(x, math.sqrt(1. / depth)))
        pyro.sample("y", Normal(x, 1.), obs=torch.tensor(float(1)))

    def factorized_guide(reparameterized):
        Normal = dist.Normal if reparameterized else dist.testing.fakes.NonreparameterizedNormal
        pyro.sample("x0", Normal(pyro.param("q2"), math.sqrt(1. / depth)))
        for i in range(1, depth):
            pyro.sample("x{}".format(i), Normal(0., math.sqrt(float(i+1) / depth)))

    def nonfactorized_guide(reparameterized):
        Normal = dist.Normal if reparameterized else dist.testing.fakes.NonreparameterizedNormal
        x = pyro.sample("x0", Normal(pyro.param("q2"), math.sqrt(1. / depth)))
        for i in range(1, depth):
            x = pyro.sample("x{}".format(i), Normal(x, math.sqrt(1. / depth)))

    with pyro_backend("contrib.funsor"):
        # compare reparameterized and nonreparameterized gradient estimates
        q2 = pyro.param("q2", torch.tensor(0.5, requires_grad=True))
        qs = (q2.unconstrained(),)

        tmc = infer.TraceTMC_ELBO(max_plate_nesting=max_plate_nesting)
        tmc_model = infer.config_enumerate(
            model, default="parallel", expand=expand, num_samples=num_samples, tmc=tmc_strategy)
        guide = factorized_guide if guide_type == "factorized" else \
            nonfactorized_guide if guide_type == "nonfactorized" else \
            lambda *args: None
        tmc_guide = infer.config_enumerate(
            guide, default="parallel", expand=expand, num_samples=num_samples, tmc=tmc_strategy)

        # convert to linear space for unbiasedness
        actual_loss = (-tmc.differentiable_loss(tmc_model, tmc_guide, reparameterized)).exp()
        actual_grads = grad(actual_loss, qs)

    # gold values from Funsor
    expected_grads = (torch.tensor(
        {1: 0.0999, 2: 0.0860, 3: 0.0802, 4: 0.0771}[depth]
    ),)

    grad_prec = 0.05 if reparameterized else 0.1

    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        print(actual_loss)
        assert_equal(actual_grad, expected_grad, prec=grad_prec, msg="".join([
            "\nexpected grad = {}".format(expected_grad.detach().cpu().numpy()),
            "\n  actual grad = {}".format(actual_grad.detach().cpu().numpy()),
        ]))


@pytest.mark.parametrize("backend", ["pyro", "contrib.funsor"])
@pytest.mark.parametrize("inner_dim", [2])
@pytest.mark.parametrize("outer_dim", [2])
def test_elbo_plate_plate(backend, outer_dim, inner_dim):
    with pyro_backend(backend):
        pyro.get_param_store().clear()
        q = pyro.param("q", torch.tensor([0.75, 0.25], requires_grad=True))
        p = 0.2693204236205713  # for which kl(Categorical(q), Categorical(p)) = 0.5
        p = torch.tensor([p, 1-p])

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
            torch.distributions.Categorical(q), torch.distributions.Categorical(p))
        kl = (1 + outer_dim + inner_dim + outer_dim * inner_dim) * kl_node
        expected_loss = kl
        expected_grad = grad(kl, [q.unconstrained()])[0]

        elbo = infer.TraceEnum_ELBO(max_plate_nesting=2)
        actual_loss = elbo.differentiable_loss(model, guide)
        actual_grad = grad(actual_loss, [q.unconstrained()])[0]

        assert_equal(actual_loss, expected_loss, prec=1e-5)
        assert_equal(actual_grad, expected_grad, prec=1e-5)


@pytest.mark.parametrize('backend', ["pyro", "contrib.funsor"])
def test_elbo_enumerate_plates_1(backend):
    #  +-----------------+
    #  | a ----> b   M=2 |
    #  +-----------------+
    #  +-----------------+
    #  | c ----> d   N=3 |
    #  +-----------------+
    # This tests two unrelated plates.
    # Each should remain uncontracted.
    with pyro_backend(backend):
        pyro.param("probs_a",
                   torch.tensor([0.45, 0.55]),
                   constraint=constraints.simplex)
        pyro.param("probs_b",
                   torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
                   constraint=constraints.simplex)
        pyro.param("probs_c",
                   torch.tensor([0.75, 0.25]),
                   constraint=constraints.simplex)
        pyro.param("probs_d",
                   torch.tensor([[0.4, 0.6], [0.3, 0.7]]),
                   constraint=constraints.simplex)
        b_data = torch.tensor([0, 1])
        d_data = torch.tensor([0, 0, 1])

        def auto_model():
            probs_a = pyro.param("probs_a")
            probs_b = pyro.param("probs_b")
            probs_c = pyro.param("probs_c")
            probs_d = pyro.param("probs_d")
            with pyro.plate("a_axis", 2, dim=-1):
                a = pyro.sample("a", dist.Categorical(probs_a),
                                infer={"enumerate": "parallel"})
                pyro.sample("b", dist.Categorical(probs_b[a]), obs=b_data)
            with pyro.plate("c_axis", 3, dim=-1):
                c = pyro.sample("c", dist.Categorical(probs_c),
                                infer={"enumerate": "parallel"})
                pyro.sample("d", dist.Categorical(probs_d[c]), obs=d_data)

        def hand_model():
            probs_a = pyro.param("probs_a")
            probs_b = pyro.param("probs_b")
            probs_c = pyro.param("probs_c")
            probs_d = pyro.param("probs_d")
            for i in range(2):
                a = pyro.sample("a_{}".format(i), dist.Categorical(probs_a),
                                infer={"enumerate": "parallel"})
                pyro.sample("b_{}".format(i), dist.Categorical(probs_b[a]), obs=b_data[i])
            for j in range(3):
                c = pyro.sample("c_{}".format(j), dist.Categorical(probs_c),
                                infer={"enumerate": "parallel"})
                pyro.sample("d_{}".format(j), dist.Categorical(probs_d[c]), obs=d_data[j])

        def guide():
            pass

        elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
        auto_loss = elbo.differentiable_loss(auto_model, guide)
        elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
        hand_loss = elbo.differentiable_loss(hand_model, guide)
        _check_loss_and_grads(hand_loss, auto_loss)


@pytest.mark.parametrize('backend', ["pyro", "contrib.funsor"])
def test_elbo_enumerate_plate_7(backend):
    #  Guide    Model
    #    a -----> b
    #    |        |
    #  +-|--------|----------------+
    #  | V        V                |
    #  | c -----> d -----> e   N=2 |
    #  +---------------------------+
    # This tests a mixture of model and guide enumeration.
    with pyro_backend(backend):
        pyro.param("model_probs_a",
                   torch.tensor([0.45, 0.55]),
                   constraint=constraints.simplex)
        pyro.param("model_probs_b",
                   torch.tensor([[0.6, 0.4], [0.4, 0.6]]),
                   constraint=constraints.simplex)
        pyro.param("model_probs_c",
                   torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
                   constraint=constraints.simplex)
        pyro.param("model_probs_d",
                   torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]]),
                   constraint=constraints.simplex)
        pyro.param("model_probs_e",
                   torch.tensor([[0.75, 0.25], [0.55, 0.45]]),
                   constraint=constraints.simplex)
        pyro.param("guide_probs_a",
                   torch.tensor([0.35, 0.64]),
                   constraint=constraints.simplex)
        pyro.param("guide_probs_c",
                   torch.tensor([[0., 1.], [1., 0.]]),  # deterministic
                   constraint=constraints.simplex)

        def auto_model(data):
            probs_a = pyro.param("model_probs_a")
            probs_b = pyro.param("model_probs_b")
            probs_c = pyro.param("model_probs_c")
            probs_d = pyro.param("model_probs_d")
            probs_e = pyro.param("model_probs_e")
            a = pyro.sample("a", dist.Categorical(probs_a))
            b = pyro.sample("b", dist.Categorical(probs_b[a]),
                            infer={"enumerate": "parallel"})
            with pyro.plate("data", 2, dim=-1):
                c = pyro.sample("c", dist.Categorical(probs_c[a]))
                d = pyro.sample("d", dist.Categorical(Vindex(probs_d)[b, c]),
                                infer={"enumerate": "parallel"})
                pyro.sample("obs", dist.Categorical(probs_e[d]), obs=data)

        def auto_guide(data):
            probs_a = pyro.param("guide_probs_a")
            probs_c = pyro.param("guide_probs_c")
            a = pyro.sample("a", dist.Categorical(probs_a),
                            infer={"enumerate": "parallel"})
            with pyro.plate("data", 2, dim=-1):
                pyro.sample("c", dist.Categorical(probs_c[a]))

        def hand_model(data):
            probs_a = pyro.param("model_probs_a")
            probs_b = pyro.param("model_probs_b")
            probs_c = pyro.param("model_probs_c")
            probs_d = pyro.param("model_probs_d")
            probs_e = pyro.param("model_probs_e")
            a = pyro.sample("a", dist.Categorical(probs_a))
            b = pyro.sample("b", dist.Categorical(probs_b[a]),
                            infer={"enumerate": "parallel"})
            for i in range(2):
                c = pyro.sample("c_{}".format(i), dist.Categorical(probs_c[a]))
                d = pyro.sample("d_{}".format(i),
                                dist.Categorical(Vindex(probs_d)[b, c]),
                                infer={"enumerate": "parallel"})
                pyro.sample("obs_{}".format(i), dist.Categorical(probs_e[d]), obs=data[i])

        def hand_guide(data):
            probs_a = pyro.param("guide_probs_a")
            probs_c = pyro.param("guide_probs_c")
            a = pyro.sample("a", dist.Categorical(probs_a),
                            infer={"enumerate": "parallel"})
            for i in range(2):
                pyro.sample("c_{}".format(i), dist.Categorical(probs_c[a]))

        data = torch.tensor([0, 0])
        elbo = infer.TraceEnum_ELBO(max_plate_nesting=1)
        auto_loss = elbo.differentiable_loss(auto_model, auto_guide, data)
        elbo = infer.TraceEnum_ELBO(max_plate_nesting=0)
        hand_loss = elbo.differentiable_loss(hand_model, hand_guide, data)
        _check_loss_and_grads(hand_loss, auto_loss)
