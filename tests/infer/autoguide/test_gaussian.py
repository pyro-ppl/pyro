# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, JitTrace_ELBO, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoGaussian, AutoGuideList
from pyro.infer.autoguide.gaussian import (
    AutoGaussianDense,
    AutoGaussianFunsor,
    _break_plates,
)
from pyro.infer.reparam import LocScaleReparam
from pyro.optim import ClippedAdam
from tests.common import assert_close, assert_equal, xfail_if_not_implemented

BACKENDS = [
    "dense",
    pytest.param("funsor", marks=[pytest.mark.stage("funsor")]),
]


def test_break_plates():
    shape = torch.Size([5, 4, 3, 2])
    x = torch.arange(shape.numel()).reshape(shape)

    MockPlate = namedtuple("MockPlate", "dim, size")
    h = MockPlate(-4, 6)
    i = MockPlate(-3, 5)
    j = MockPlate(-2, 4)
    k = MockPlate(-1, 3)

    actual = _break_plates(x, {i, j, k}, set())
    expected = x.reshape(-1)
    assert_equal(actual, expected)

    actual = _break_plates(x, {i, j, k}, {i})
    expected = x.reshape(5, 1, 1, -1)
    assert_equal(actual, expected)

    actual = _break_plates(x, {i, j, k}, {j})
    expected = x.permute((1, 0, 2, 3)).reshape(4, 1, -1)
    assert_equal(actual, expected)

    actual = _break_plates(x, {i, j, k}, {k})
    expected = x.permute((2, 0, 1, 3)).reshape(3, -1)
    assert_equal(actual, expected)

    actual = _break_plates(x, {i, j, k}, {i, j})
    expected = x.reshape(5, 4, 1, -1)
    assert_equal(actual, expected)

    actual = _break_plates(x, {i, j, k}, {i, k})
    expected = x.permute((0, 2, 1, 3)).reshape(5, 1, 3, -1)
    assert_equal(actual, expected)

    actual = _break_plates(x, {i, j, k}, {j, k})
    expected = x.permute((1, 2, 0, 3)).reshape(4, 3, -1)
    assert_equal(actual, expected)

    actual = _break_plates(x, {i, j, k}, {i, j, k})
    expected = x
    assert_equal(actual, expected)

    actual = _break_plates(x, {i, j, k}, {h, i, j, k})
    expected = x
    assert_equal(actual, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_dispatch(backend):
    def model():
        pyro.sample("x", dist.Normal(0, 1))

    guide = AutoGaussian(model, backend=backend)
    if backend == "dense":
        assert isinstance(guide, AutoGaussianDense)
        guide = AutoGaussianDense(model)
        assert isinstance(guide, AutoGaussianDense)
    elif backend == "funsor":
        assert isinstance(guide, AutoGaussianFunsor)
        guide = AutoGaussianFunsor(model)
        assert isinstance(guide, AutoGaussianFunsor)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def check_structure(model, expected_str, expected_dependencies=None):
    guide = AutoGaussian(model, backend="dense")
    guide()  # initialize
    if expected_dependencies is not None:
        assert guide.dependencies == expected_dependencies

    # Inject random noise into all unconstrained parameters.
    for parameter in guide.parameters():
        parameter.data.normal_()

    with torch.no_grad():
        # Check flatten & unflatten.
        mvn = guide._dense_get_mvn()
        expected = mvn.sample()
        samples = guide._dense_unflatten(expected)
        actual = guide._dense_flatten(samples)
        assert_equal(actual, expected)

        # Check sparsity structure.
        precision = mvn.precision_matrix
        actual = precision.abs().gt(1e-5).long()
        str_to_number = {"?": 1, ".": 0}
        expected = torch.tensor(
            [[str_to_number[c] for c in row if c != " "] for row in expected_str]
        )
        assert (actual == expected).all()


def check_backends_agree(model):
    guide1 = AutoGaussian(model, backend="dense")
    guide2 = AutoGaussian(model, backend="funsor")
    guide1()
    with xfail_if_not_implemented():
        guide2()

    # Inject random noise into all unconstrained parameters.
    params1 = dict(guide1.named_parameters())
    params2 = dict(guide2.named_parameters())
    assert set(params1) == set(params2)
    for k, v in params1.items():
        v.data.add_(torch.zeros_like(v).normal_())
        params2[k].data.copy_(v.data)
    names = sorted(params1)

    # Check densities agree between backends.
    with torch.no_grad(), poutine.trace() as tr:
        aux = guide2._sample_aux_values(temperature=1.0)
        flat = guide1._dense_flatten(aux)
        tr.trace.compute_log_prob()
    log_prob_funsor = tr.trace.nodes["_AutoGaussianFunsor_latent"]["log_prob"]
    with torch.no_grad(), poutine.trace() as tr:
        with poutine.condition(data={"_AutoGaussianDense_latent": flat}):
            guide1._sample_aux_values(temperature=1.0)
        tr.trace.compute_log_prob()
    log_prob_dense = tr.trace.nodes["_AutoGaussianDense_latent"]["log_prob"]
    assert_equal(log_prob_funsor, log_prob_dense)

    # Check Monte Carlo estimate of entropy.
    entropy1 = guide1._dense_get_mvn().entropy()
    with pyro.plate("particle", 100000, dim=-3), poutine.trace() as tr:
        guide2._sample_aux_values(temperature=1.0)
        tr.trace.compute_log_prob()
    entropy2 = -tr.trace.nodes["_AutoGaussianFunsor_latent"]["log_prob"].mean()
    assert_close(entropy1, entropy2, atol=1e-2)
    grads1 = torch.autograd.grad(
        entropy1, [params1[k] for k in names], allow_unused=True
    )
    grads2 = torch.autograd.grad(
        entropy2, [params2[k] for k in names], allow_unused=True
    )
    for name, grad1, grad2 in zip(names, grads1, grads2):
        # Gradients should agree to very high precision.
        if grad1 is None and grad2 is not None:
            grad1 = torch.zeros_like(grad2)
        elif grad2 is None and grad1 is not None:
            grad2 = torch.zeros_like(grad1)
        assert_close(grad1, grad2, msg=f"{name}:\n{grad1} vs {grad2}")

    # Check elbos agree between backends.
    elbo = Trace_ELBO(num_particles=1000000, vectorize_particles=True)
    loss1 = elbo.differentiable_loss(model, guide1)
    loss2 = elbo.differentiable_loss(model, guide2)
    assert_close(loss1, loss2, atol=1e-2, rtol=0.05)
    grads1 = torch.autograd.grad(loss1, [params1[k] for k in names], allow_unused=True)
    grads2 = torch.autograd.grad(loss2, [params2[k] for k in names], allow_unused=True)
    for name, grad1, grad2 in zip(names, grads1, grads2):
        assert_close(
            grad1, grad2, atol=0.05, rtol=0.05, msg=f"{name}:\n{grad1} vs {grad2}"
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_structure_0(backend):
    def model():
        a = pyro.sample("a", dist.Normal(0, 1))
        pyro.sample("b", dist.Normal(a, 1), obs=torch.ones(()))

    # size = 1
    structure = [
        "?",
    ]
    dependencies = {
        "a": {"a": set()},
        "b": {"b": set(), "a": set()},
    }
    if backend == "funsor":
        check_backends_agree(model)
    else:
        check_structure(model, structure, dependencies)


@pytest.mark.parametrize("backend", BACKENDS)
def test_structure_1(backend):
    def model():
        a = pyro.sample("a", dist.Normal(0, 1))
        with pyro.plate("i", 3):
            pyro.sample("b", dist.Normal(a, 1), obs=torch.ones(3))

    # size = 1
    structure = [
        "?",
    ]
    dependencies = {
        "a": {"a": set()},
        "b": {"b": set(), "a": set()},
    }
    if backend == "funsor":
        check_backends_agree(model)
    else:
        check_structure(model, structure, dependencies)


@pytest.mark.parametrize("backend", BACKENDS)
def test_structure_2(backend):
    def model():
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(a, 1))
        c = pyro.sample("c", dist.Normal(b, 1))
        pyro.sample("d", dist.Normal(c, 1), obs=torch.tensor(1.0))

    # size = 1 + 1 + 1 = 3
    structure = [
        "? ? .",
        "? ? ?",
        ". ? ?",
    ]
    dependencies = {
        "a": {"a": set()},
        "b": {"b": set(), "a": set()},
        "c": {"c": set(), "b": set()},
        "d": {"c": set(), "d": set()},
    }
    if backend == "funsor":
        check_backends_agree(model)
    else:
        check_structure(model, structure, dependencies)


@pytest.mark.parametrize("backend", BACKENDS)
def test_structure_3(backend):
    def model():
        with pyro.plate("i", 2):
            a = pyro.sample("a", dist.Normal(0, 1))
            b = pyro.sample("b", dist.Normal(a, 1))
            c = pyro.sample("c", dist.Normal(b, 1))
            pyro.sample("d", dist.Normal(c, 1), obs=torch.tensor(1.0))

    # size = 2 + 2 + 2 = 6
    structure = [
        "? . ? . . .",
        ". ? . ? . .",
        "? . ? . ? .",
        ". ? . ? . ?",
        ". . ? . ? .",
        ". . . ? . ?",
    ]
    dependencies = {
        "a": {"a": set()},
        "b": {"b": set(), "a": set()},
        "c": {"c": set(), "b": set()},
        "d": {"c": set(), "d": set()},
    }
    if backend == "funsor":
        check_backends_agree(model)
    else:
        check_structure(model, structure, dependencies)


@pytest.mark.parametrize("backend", BACKENDS)
def test_structure_4(backend):
    def model():
        a = pyro.sample("a", dist.Normal(0, 1))
        with pyro.plate("i", 2):
            b = pyro.sample("b", dist.Normal(a, 1))
            c = pyro.sample("c", dist.Normal(b, 1))
        pyro.sample("d", dist.Normal(c.sum(), 1), obs=torch.tensor(1.0))

    # size = 1 + 2 + 2 = 5
    structure = [
        "? ? ? . .",
        "? ? . ? .",
        "? . ? . ?",
        ". ? . ? ?",
        ". . ? ? ?",
    ]
    dependencies = {
        "a": {"a": set()},
        "b": {"b": set(), "a": set()},
        "c": {"c": set(), "b": set()},
        "d": {"c": set(), "d": set()},
    }
    if backend == "funsor":
        check_backends_agree(model)
    else:
        check_structure(model, structure, dependencies)


@pytest.mark.parametrize("backend", BACKENDS)
def test_structure_5(backend):
    def model():
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(0, 1))
        with pyro.plate("i", 2):
            c = pyro.sample("c", dist.Normal(a, b.exp()))
            pyro.sample("d", dist.Normal(c, 1), obs=torch.tensor(1.0))

    # size = 1 + 1 + 2 = 4
    structure = [
        "? ? ? ?",
        "? ? ? ?",
        "? ? ? .",
        "? ? . ?",
    ]
    if backend == "funsor":
        check_backends_agree(model)
    else:
        check_structure(model, structure)


@pytest.mark.parametrize("backend", BACKENDS)
def test_structure_6(backend):
    I, J = 2, 3

    def model():
        i_plate = pyro.plate("i", I, dim=-1)
        j_plate = pyro.plate("j", J, dim=-2)
        with i_plate:
            w = pyro.sample("w", dist.Normal(0, 1))
        with j_plate:
            x = pyro.sample("x", dist.Normal(0, 1))
        with i_plate, j_plate:
            y = pyro.sample("y", dist.Normal(w, x.exp()))
            pyro.sample("z", dist.Normal(1, 1), obs=y)

    # size = 2 + 3 + 2 * 3 = 2 + 3 + 6 = 11
    structure = [
        "? . ? ? ? ? . ? . ? .",
        ". ? ? ? ? . ? . ? . ?",
        "? ? ? . . ? ? . . . .",
        "? ? . ? . . . ? ? . .",
        "? ? . . ? . . . . ? ?",
        "? . ? . . ? . . . . .",
        ". ? ? . . . ? . . . .",
        "? . . ? . . . ? . . .",
        ". ? . ? . . . . ? . .",
        "? . . . ? . . . . ? .",
        ". ? . . ? . . . . . ?",
    ]
    if backend == "funsor":
        check_backends_agree(model)
    else:
        check_structure(model, structure)


@pytest.mark.parametrize("backend", BACKENDS)
def test_structure_7(backend):
    I, J = 2, 3

    def model():
        i_plate = pyro.plate("i", I, dim=-1)
        j_plate = pyro.plate("j", J, dim=-2)
        a = pyro.sample("a", dist.Normal(0, 1))
        with i_plate:
            b = pyro.sample("b", dist.Normal(a, 1))
        with j_plate:
            c = pyro.sample("c", dist.Normal(b.mean(), 1))
        d = pyro.sample("d", dist.Normal(c.mean(), 1))
        pyro.sample("e", dist.Normal(1, 1), obs=d)

    # size = 1 + 2 + 3 + 1 = 7
    structure = [
        "? ? ? . . . .",
        "? ? ? ? ? ? .",
        "? ? ? ? ? ? .",
        ". ? ? ? ? ? ?",
        ". ? ? ? ? ? ?",
        ". ? ? ? ? ? ?",
        ". . . ? ? ? ?",
    ]
    if backend == "funsor":
        check_backends_agree(model)
    else:
        check_structure(model, structure)


@pytest.mark.parametrize("backend", BACKENDS)
def test_structure_8(backend):
    def model():
        i_plate = pyro.plate("i", 2, dim=-1)
        with i_plate:
            a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(a.mean(-1), 1))
        with i_plate:
            pyro.sample("c", dist.Normal(b, 1), obs=torch.ones(2))

    # size = 2 + 1 = 3
    structure = [
        "? ? ?",
        "? ? ?",
        "? ? ?",
    ]
    if backend == "funsor":
        check_backends_agree(model)
    else:
        check_structure(model, structure)


@pytest.mark.parametrize("backend", BACKENDS)
def test_broken_plates_smoke(backend):
    def model():
        with pyro.plate("i", 2):
            a = pyro.sample("a", dist.Normal(0, 1))
        pyro.sample("b", dist.Normal(a.mean(-1), 1), obs=torch.tensor(0.0))

    guide = AutoGaussian(model, backend=backend)
    svi = SVI(model, guide, ClippedAdam({"lr": 1e-8}), Trace_ELBO())
    for step in range(2):
        with xfail_if_not_implemented():
            svi.step()
    guide()
    predictive = Predictive(model, guide=guide, num_samples=2)
    predictive()


@pytest.mark.parametrize("backend", BACKENDS)
def test_intractable_smoke(backend):
    def model():
        i_plate = pyro.plate("i", 2, dim=-1)
        j_plate = pyro.plate("j", 3, dim=-2)
        with i_plate:
            a = pyro.sample("a", dist.Normal(0, 1))
        with j_plate:
            b = pyro.sample("b", dist.Normal(0, 1))
        with i_plate, j_plate:
            c = pyro.sample("c", dist.Normal(a + b, 1))
            pyro.sample("d", dist.Normal(c, 1), obs=torch.zeros(3, 2))

    guide = AutoGaussian(model, backend=backend)
    svi = SVI(model, guide, ClippedAdam({"lr": 1e-8}), Trace_ELBO())
    for step in range(2):
        with xfail_if_not_implemented():
            svi.step()
    guide()
    predictive = Predictive(model, guide=guide, num_samples=2)
    predictive()


# Simplified from https://github.com/pyro-cov/tree/master/pyrocov/mutrans.py
def pyrocov_model(dataset):
    # Tensor shapes are commented at the end of some lines.
    features = dataset["features"]
    local_time = dataset["local_time"][..., None]  # [T, P, 1]
    T, P, _ = local_time.shape
    S, F = features.shape
    weekly_strains = dataset["weekly_strains"]
    assert weekly_strains.shape == (T, P, S)

    # Sample global random variables.
    coef_scale = pyro.sample("coef_scale", dist.InverseGamma(5e3, 1e2))[..., None]
    rate_scale = pyro.sample("rate_scale", dist.LogNormal(-4, 2))[..., None]
    init_loc_scale = pyro.sample("init_loc_scale", dist.LogNormal(0, 2))[..., None]
    init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))[..., None]

    # Assume relative growth rate depends strongly on mutations and weakly on place.
    coef_loc = torch.zeros(F)
    coef = pyro.sample("coef", dist.Logistic(coef_loc, coef_scale).to_event(1))  # [F]
    rate_loc = pyro.deterministic(
        "rate_loc", 0.01 * coef @ features.T, event_dim=1
    )  # [S]

    # Assume initial infections depend strongly on strain and place.
    init_loc = pyro.sample(
        "init_loc", dist.Normal(torch.zeros(S), init_loc_scale).to_event(1)
    )  # [S]
    with pyro.plate("place", P, dim=-1):
        rate = pyro.sample(
            "rate", dist.Normal(rate_loc, rate_scale).to_event(1)
        )  # [P, S]
        init = pyro.sample(
            "init", dist.Normal(init_loc, init_scale).to_event(1)
        )  # [P, S]

        # Finally observe counts.
        with pyro.plate("time", T, dim=-2):
            logits = init + rate * local_time  # [T, P, S]
            pyro.sample(
                "obs",
                dist.Multinomial(logits=logits, validate_args=False),
                obs=weekly_strains,
            )


# This is modified by relaxing rate from deterministic to latent.
def pyrocov_model_relaxed(dataset):
    # Tensor shapes are commented at the end of some lines.
    features = dataset["features"]
    local_time = dataset["local_time"][..., None]  # [T, P, 1]
    T, P, _ = local_time.shape
    S, F = features.shape
    weekly_strains = dataset["weekly_strains"]
    assert weekly_strains.shape == (T, P, S)

    # Sample global random variables.
    coef_scale = pyro.sample("coef_scale", dist.InverseGamma(5e3, 1e2))[..., None]
    rate_loc_scale = pyro.sample("rate_loc_scale", dist.LogNormal(-4, 2))[..., None]
    rate_scale = pyro.sample("rate_scale", dist.LogNormal(-4, 2))[..., None]
    init_loc_scale = pyro.sample("init_loc_scale", dist.LogNormal(0, 2))[..., None]
    init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))[..., None]

    # Assume relative growth rate depends strongly on mutations and weakly on place.
    coef_loc = torch.zeros(F)
    coef = pyro.sample("coef", dist.Logistic(coef_loc, coef_scale).to_event(1))  # [F]
    rate_loc = pyro.sample(
        "rate_loc",
        dist.Normal(0.01 * coef @ features.T, rate_loc_scale).to_event(1),
    )  # [S]

    # Assume initial infections depend strongly on strain and place.
    init_loc = pyro.sample(
        "init_loc", dist.Normal(torch.zeros(S), init_loc_scale).to_event(1)
    )  # [S]
    with pyro.plate("place", P, dim=-1):
        rate = pyro.sample(
            "rate", dist.Normal(rate_loc, rate_scale).to_event(1)
        )  # [P, S]
        init = pyro.sample(
            "init", dist.Normal(init_loc, init_scale).to_event(1)
        )  # [P, S]

        # Finally observe counts.
        with pyro.plate("time", T, dim=-2):
            logits = init + rate * local_time  # [T, P, S]
            pyro.sample(
                "obs",
                dist.Multinomial(logits=logits, validate_args=False),
                obs=weekly_strains,
            )


# This is modified by more precisely tracking plates for features and strains.
def pyrocov_model_plated(dataset):
    # Tensor shapes are commented at the end of some lines.
    features = dataset["features"]
    local_time = dataset["local_time"][..., None]  # [T, P, 1]
    T, P, _ = local_time.shape
    S, F = features.shape
    weekly_strains = dataset["weekly_strains"]  # [T, P, S]
    assert weekly_strains.shape == (T, P, S)
    feature_plate = pyro.plate("feature", F, dim=-1)
    strain_plate = pyro.plate("strain", S, dim=-1)
    place_plate = pyro.plate("place", P, dim=-2)
    time_plate = pyro.plate("time", T, dim=-3)

    # Sample global random variables.
    coef_scale = pyro.sample("coef_scale", dist.InverseGamma(5e3, 1e2))
    rate_loc_scale = pyro.sample("rate_loc_scale", dist.LogNormal(-4, 2))
    rate_scale = pyro.sample("rate_scale", dist.LogNormal(-4, 2))
    init_loc_scale = pyro.sample("init_loc_scale", dist.LogNormal(0, 2))
    init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))

    with feature_plate:
        coef = pyro.sample("coef", dist.Logistic(0, coef_scale))  # [F]
    rate_loc_loc = 0.01 * coef @ features.T
    with strain_plate:
        rate_loc = pyro.sample(
            "rate_loc", dist.Normal(rate_loc_loc, rate_loc_scale)
        )  # [S]
        init_loc = pyro.sample("init_loc", dist.Normal(0, init_loc_scale))  # [S]
    with place_plate, strain_plate:
        rate = pyro.sample("rate", dist.Normal(rate_loc, rate_scale))  # [P, S]
        init = pyro.sample("init", dist.Normal(init_loc, init_scale))  # [P, S]

    # Finally observe counts.
    with time_plate, place_plate:
        logits = (init + rate * local_time)[..., None, :]  # [T, P, 1, S]
        pyro.sample(
            "obs",
            dist.Multinomial(logits=logits, validate_args=False),
            obs=weekly_strains[..., None, :],
        )


# This is modified by replacing the multinomial likelihood with poisson.
def pyrocov_model_poisson(dataset):
    # Tensor shapes are commented at the end of some lines.
    features = dataset["features"]
    local_time = dataset["local_time"][..., None]  # [T, P, 1]
    T, P, _ = local_time.shape
    S, F = features.shape
    weekly_strains = dataset["weekly_strains"]  # [T, P, S]
    if not torch._C._get_tracing_state():
        assert weekly_strains.shape == (T, P, S)
    strain_plate = pyro.plate("strain", S, dim=-1)
    place_plate = pyro.plate("place", P, dim=-2)
    time_plate = pyro.plate("time", T, dim=-3)

    # Sample global random variables.
    coef_scale = pyro.sample("coef_scale", dist.LogNormal(-4, 2))
    rate_loc_scale = pyro.sample("rate_loc_scale", dist.LogNormal(-4, 2))
    rate_scale = pyro.sample("rate_scale", dist.LogNormal(-4, 2))
    init_loc_scale = pyro.sample("init_loc_scale", dist.LogNormal(0, 2))
    init_scale = pyro.sample("init_scale", dist.LogNormal(0, 2))
    pois_loc = pyro.sample("pois_loc", dist.Normal(0, 2))
    pois_scale = pyro.sample("pois_scale", dist.LogNormal(0, 2))

    coef = pyro.sample(
        "coef", dist.Logistic(torch.zeros(F), coef_scale).to_event(1)
    )  # [F]
    rate_loc_loc = 0.01 * coef @ features.T
    with strain_plate:
        rate_loc = pyro.sample(
            "rate_loc", dist.Normal(rate_loc_loc, rate_loc_scale)
        )  # [S]
        init_loc = pyro.sample("init_loc", dist.Normal(0, init_loc_scale))  # [S]
    with place_plate, strain_plate:
        rate = pyro.sample("rate", dist.Normal(rate_loc, rate_scale))  # [P, S]
        init = pyro.sample("init", dist.Normal(init_loc, init_scale))  # [P, S]

    # Finally observe counts.
    with time_plate, place_plate:
        pois = pyro.sample("pois", dist.LogNormal(pois_loc, pois_scale))
    with time_plate, place_plate, strain_plate:
        # Note .softmax() breaks conditional independence over strain, but only
        # weakly. We could directly call .exp(), but .softmax is more
        # numerically stable.
        logits = pois * (init + rate * local_time).softmax(-1)  # [T, P, S]
        pyro.sample("obs", dist.Poisson(logits), obs=weekly_strains)


class PoissonGuide(AutoGuideList):
    def __init__(self, model, backend):
        super().__init__(model)
        self.append(
            AutoGaussian(poutine.block(model, hide_fn=self.hide_fn_1), backend=backend)
        )
        self.append(
            AutoGaussian(poutine.block(model, hide_fn=self.hide_fn_2), backend=backend)
        )

    @staticmethod
    def hide_fn_1(msg):
        return msg["type"] == "sample" and "pois" in msg["name"]

    @staticmethod
    def hide_fn_2(msg):
        return msg["type"] == "sample" and "pois" not in msg["name"]


PYRO_COV_MODELS = [
    (pyrocov_model, AutoGaussian),
    (pyrocov_model_relaxed, AutoGaussian),
    (pyrocov_model_plated, AutoGaussian),
    (pyrocov_model_poisson, PoissonGuide),
]


@pytest.mark.parametrize("model, Guide", PYRO_COV_MODELS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_pyrocov_smoke(model, Guide, backend):
    T, P, S, F = 3, 4, 5, 6
    dataset = {
        "features": torch.randn(S, F),
        "local_time": torch.randn(T, P),
        "weekly_strains": torch.randn(T, P, S).exp().round(),
    }

    guide = Guide(model, backend=backend)
    svi = SVI(model, guide, ClippedAdam({"lr": 1e-8}), Trace_ELBO())
    for step in range(2):
        with xfail_if_not_implemented():
            svi.step(dataset)
    guide(dataset)
    predictive = Predictive(model, guide=guide, num_samples=2)
    predictive(dataset)


@pytest.mark.parametrize("model, Guide", PYRO_COV_MODELS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_pyrocov_reparam(model, Guide, backend):
    T, P, S, F = 2, 3, 4, 5
    dataset = {
        "features": torch.randn(S, F),
        "local_time": torch.randn(T, P),
        "weekly_strains": torch.randn(T, P, S).exp().round(),
    }

    # Reparametrize the model.
    config = {
        "coef": LocScaleReparam(),
        "rate_loc": None if model is pyrocov_model else LocScaleReparam(),
        "rate": LocScaleReparam(),
        "init_loc": LocScaleReparam(),
        "init": LocScaleReparam(),
    }
    model = poutine.reparam(model, config)
    guide = Guide(model, backend=backend)
    svi = SVI(model, guide, ClippedAdam({"lr": 1e-8}), Trace_ELBO())
    for step in range(2):
        with xfail_if_not_implemented():
            svi.step(dataset)
    guide(dataset)
    predictive = Predictive(model, guide=guide, num_samples=2)
    predictive(dataset)


@pytest.mark.stage("funsor")
def test_pyrocov_structure():
    from funsor import Bint, Real, Reals

    T, P, S, F = 2, 3, 4, 5
    dataset = {
        "features": torch.randn(S, F),
        "local_time": torch.randn(T, P),
        "weekly_strains": torch.randn(T, P, S).exp().round(),
    }

    guide = PoissonGuide(pyrocov_model_poisson, backend="funsor")
    guide(dataset)  # initialize
    guide = guide[0]  # pull out AutoGaussian part of PoissonGuide

    expected_plates = frozenset(["place", "strain"])
    assert guide._funsor_plates == expected_plates

    expected_eliminate = frozenset(
        [
            "coef",
            "coef_scale",
            "init",
            "init_loc",
            "init_loc_scale",
            "init_scale",
            "place",
            "rate",
            "rate_loc",
            "rate_loc_scale",
            "rate_scale",
            "strain",
        ]
    )
    assert guide._funsor_eliminate == expected_eliminate

    expected_factor_inputs = {
        "coef_scale": OrderedDict([("coef_scale", Real)]),
        "rate_loc_scale": OrderedDict([("rate_loc_scale", Real)]),
        "rate_scale": OrderedDict([("rate_scale", Real)]),
        "init_loc_scale": OrderedDict([("init_loc_scale", Real)]),
        "init_scale": OrderedDict([("init_scale", Real)]),
        "coef": OrderedDict([("coef", Reals[5]), ("coef_scale", Real)]),
        "rate_loc": OrderedDict(
            [
                ("strain", Bint[4]),
                ("rate_loc", Real),
                ("rate_loc_scale", Real),
                ("coef", Reals[5]),
            ]
        ),
        "init_loc": OrderedDict(
            [
                ("strain", Bint[4]),
                ("init_loc", Real),
                ("init_loc_scale", Real),
            ]
        ),
        "rate": OrderedDict(
            [
                ("place", Bint[3]),
                ("strain", Bint[4]),
                ("rate", Real),
                ("rate_scale", Real),
                ("rate_loc", Real),
            ]
        ),
        "init": OrderedDict(
            [
                ("place", Bint[3]),
                ("strain", Bint[4]),
                ("init", Real),
                ("init_scale", Real),
                ("init_loc", Real),
            ]
        ),
        "obs": OrderedDict(
            [
                ("place", Bint[3]),
                ("strain", Bint[4]),
                ("rate", Real),
                ("init", Real),
            ]
        ),
    }
    assert guide._funsor_factor_inputs == expected_factor_inputs


@pytest.mark.parametrize("jit", [False, True], ids=["nojit", "jit"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_profile(backend, jit, n=1, num_steps=1, log_every=1):
    """
    Helper function for profiling.
    """
    print("Generating fake data")
    model = pyrocov_model_poisson
    T, P, S, F = min(n, 50), n + 1, n + 2, n + 3
    dataset = {
        "features": torch.randn(S, F),
        "local_time": torch.randn(T, P),
        "weekly_strains": torch.randn(T, P, S).exp().round(),
    }

    print("Initializing guide")
    guide = PoissonGuide(model, backend=backend)
    guide(dataset)  # initialize
    print("Parameter shapes:")
    for name, param in guide.named_parameters():
        print(f"  {name}: {tuple(param.shape)}")

    print("Training")
    Elbo = JitTrace_ELBO if jit else Trace_ELBO
    elbo = Elbo(max_plate_nesting=3, ignore_jit_warnings=True)
    svi = SVI(model, guide, ClippedAdam({"lr": 1e-8}), elbo)
    for step in range(num_steps):
        loss = svi.step(dataset)
        if log_every and step % log_every == 0:
            print(f"step {step} loss = {loss}")


if __name__ == "__main__":
    import argparse
    import cProfile

    # Usage: time python -m tests.infer.autoguide.test_autoguide
    parser = argparse.ArgumentParser(description="Profiler for pyro-cov model")
    parser.add_argument("-b", "--backend", default="funsor")
    parser.add_argument("-s", "--size", default=10, type=int)
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-fp64", "--double", action="store_true")
    parser.add_argument("-fp32", "--float", action="store_false", dest="double")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cpu", dest="cuda", action="store_false")
    parser.add_argument("--jit", default=True, action="store_true")
    parser.add_argument("--no-jit", dest="jit", action="store_false")
    parser.add_argument("-l", "--log-every", default=1, type=int)
    parser.add_argument("-p", "--profile")
    args = parser.parse_args()

    torch.set_default_dtype(torch.double if args.double else torch.float)
    if args.cuda:
        torch.set_default_device("cuda")

    if args.profile:
        p = cProfile.Profile()
        p.enable()
    test_profile(
        backend=args.backend,
        jit=args.jit,
        n=args.size,
        num_steps=args.num_steps,
        log_every=args.log_every,
    )
    if args.profile:
        p.disable()
        p.dump_stats(args.profile)
