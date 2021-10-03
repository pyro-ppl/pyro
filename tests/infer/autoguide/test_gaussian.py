# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict, namedtuple

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoGaussian
from pyro.infer.autoguide.gaussian import _break_plates
from pyro.infer.reparam import LocScaleReparam
from pyro.optim import Adam
from tests.common import assert_equal, xfail_if_not_implemented

BACKENDS = [
    "dense",
    pytest.param("funsor", marks=[pytest.mark.stage("funsor")]),
]


MockPlate = namedtuple("MockPlate", "dim, size")


def test_break_plates():
    shape = torch.Size([5, 4, 3, 2])
    h = MockPlate(-4, 6)
    i = MockPlate(-3, 5)
    j = MockPlate(-2, 4)
    k = MockPlate(-1, 3)
    x = torch.arange(shape.numel()).reshape(shape)

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


def check_structure(model, expected_str):
    guide = AutoGaussian(model, backend="dense")
    guide()  # initialize

    # Inject random noise into all unconstrained parameters.
    for parameter in guide.parameters():
        parameter.data.normal_()

    with torch.no_grad():
        precision = guide._get_precision()
        actual = precision.abs().gt(1e-5).long()

    str_to_number = {"?": 1, ".": 0}
    expected = torch.tensor(
        [[str_to_number[c] for c in row if c != " "] for row in expected_str]
    )
    assert_equal(actual, expected)


def test_structure_1():
    def model():
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(a, 1))
        c = pyro.sample("c", dist.Normal(b, 1))
        pyro.sample("d", dist.Normal(c, 1), obs=torch.tensor(0.0))

    expected = [
        "? ? .",
        "? ? ?",
        ". ? ?",
    ]
    check_structure(model, expected)


def test_structure_2():
    def model():
        a = pyro.sample("a", dist.Normal(0, 1))
        b = pyro.sample("b", dist.Normal(0, 1))
        with pyro.plate("i", 2):
            c = pyro.sample("c", dist.Normal(a, b.exp()))
            pyro.sample("d", dist.Normal(c, 1), obs=torch.tensor(0.0))

    # size = 1 + 1 + 2 = 4
    expected = [
        "? . ? ?",
        ". ? ? ?",
        "? ? ? .",
        "? ? . ?",
    ]
    check_structure(model, expected)


def test_structure_3():
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
            pyro.sample("z", dist.Normal(0, 1), obs=y)

    # size = 2 + 3 + 2 * 3 = 2 + 3 + 6 = 11
    expected = [
        "? . . . . ? . ? . ? .",
        ". ? . . . . ? . ? . ?",
        ". . ? . . ? ? . . . .",
        ". . . ? . . . ? ? . .",
        ". . . . ? . . . . ? ?",
        "? . ? . . ? . . . . .",
        ". ? ? . . . ? . . . .",
        "? . . ? . . . ? . . .",
        ". ? . ? . . . . ? . .",
        "? . . . ? . . . . ? .",
        ". ? . . ? . . . . . ?",
    ]
    check_structure(model, expected)


def test_structure_4():
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
        pyro.sample("e", dist.Normal(0, 1), obs=d)

    # size = 1 + 2 + 3 + 1 = 7
    expected = [
        "? ? ? . . . .",
        "? ? . ? ? ? .",
        "? . ? ? ? ? .",
        ". ? ? ? . . ?",
        ". ? ? . ? . ?",
        ". ? ? . . ? ?",
        ". . . ? ? ? ?",
    ]
    check_structure(model, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_broken_plates_smoke(backend):
    def model():
        with pyro.plate("i", 2):
            x = pyro.sample("x", dist.Normal(0, 1))
        pyro.sample("y", dist.Normal(x.mean(-1), 1), obs=torch.tensor(0.0))

    guide = AutoGaussian(model, backend=backend)
    svi = SVI(model, guide, Adam({"lr": 1e-8}), Trace_ELBO())
    for step in range(2):
        with xfail_if_not_implemented():
            svi.step()
    guide()
    predictive = Predictive(model, guide=guide, num_samples=2)
    predictive()


@pytest.mark.parametrize("backend", BACKENDS)
def test_intractable_smoke(backend):
    def model():
        with pyro.plate("i", 2):
            x = pyro.sample("x", dist.Normal(0, 1))
        pyro.sample("y", dist.Normal(x.mean(-1), 1), obs=torch.tensor(0.0))

    guide = AutoGaussian(model, backend=backend)
    svi = SVI(model, guide, Adam({"lr": 1e-8}), Trace_ELBO())
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
    pois_loc = pyro.sample("pois_loc", dist.Normal(0, 2))
    pois_scale = pyro.sample("pois_scale", dist.LogNormal(0, 2))

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
        pois = pyro.sample("pois", dist.LogNormal(pois_loc, pois_scale))
    with time_plate, place_plate, strain_plate:
        # Note .softmax() breaks conditional independence over strain, but only
        # weakly. We could directly call .exp(), but .softmax is more
        # numerically stable.
        logits = pois * (init + rate * local_time).softmax(-1)  # [T, P, S]
        pyro.sample("obs", dist.Poisson(logits), obs=weekly_strains)


PYRO_COV_MODELS = [
    pyrocov_model,
    pyrocov_model_relaxed,
    pyrocov_model_plated,
    pyrocov_model_poisson,
]


@pytest.mark.parametrize("model", PYRO_COV_MODELS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_pyrocov_smoke(model, backend):
    T, P, S, F = 3, 4, 5, 6
    dataset = {
        "features": torch.randn(S, F),
        "local_time": torch.randn(T, P),
        "weekly_strains": torch.randn(T, P, S).exp().round(),
    }

    guide = AutoGaussian(model, backend=backend)
    svi = SVI(model, guide, Adam({"lr": 1e-8}), Trace_ELBO())
    for step in range(2):
        svi.step(dataset)
    guide(dataset)
    predictive = Predictive(model, guide=guide, num_samples=2)
    predictive(dataset)


@pytest.mark.parametrize("model", PYRO_COV_MODELS)
@pytest.mark.parametrize("backend", BACKENDS)
def test_pyrocov_reparam(model, backend):
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
    guide = AutoGaussian(model, backend=backend)
    svi = SVI(model, guide, Adam({"lr": 1e-8}), Trace_ELBO())
    for step in range(2):
        svi.step(dataset)
    guide(dataset)
    predictive = Predictive(model, guide=guide, num_samples=2)
    predictive(dataset)


@pytest.mark.parametrize("backend", ["funsor"])
def test_pyrocov_structure(backend):
    from funsor import Bint, Real, Reals

    T, P, S, F = 2, 3, 4, 5
    dataset = {
        "features": torch.randn(S, F),
        "local_time": torch.randn(T, P),
        "weekly_strains": torch.randn(T, P, S).exp().round(),
    }

    guide = AutoGaussian(pyrocov_model_plated, backend=backend)
    guide(dataset)  # initialize

    expected_plates = frozenset(["place", "feature", "strain"])
    assert guide._funsor_plates == expected_plates

    expected_eliminate = frozenset(
        [
            "place",
            "coef_scale",
            "rate_loc_scale",
            "rate_scale",
            "init_loc_scale",
            "init_scale",
            "coef",
            "rate_loc",
            "init_loc",
            "rate",
            "init",
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
            [("rate_loc", Reals[4]), ("rate_loc_scale", Real), ("coef", Reals[5])]
        ),
        "init_loc": OrderedDict([("init_loc", Reals[4]), ("init_loc_scale", Real)]),
        "rate": OrderedDict(
            [
                ("place", Bint[3]),
                ("rate", Reals[4]),
                ("rate_scale", Real),
                ("rate_loc", Reals[4]),
            ]
        ),
        "init": OrderedDict(
            [
                ("place", Bint[3]),
                ("init", Reals[4]),
                ("init_scale", Real),
                ("init_loc", Reals[4]),
            ]
        ),
    }
    assert guide._funsor_factor_inputs == expected_factor_inputs


@pytest.mark.parametrize("backend", BACKENDS)
def test_profile(backend, n=1, num_steps=1):
    """
    Helper function for profiling.
    """
    model = pyrocov_model_poisson
    T, P, S, F = 2 * n, 3 * n, 4 * n, 5 * n
    dataset = {
        "features": torch.randn(S, F),
        "local_time": torch.randn(T, P),
        "weekly_strains": torch.randn(T, P, S).exp().round(),
    }

    guide = AutoGaussian(model)
    svi = SVI(model, guide, Adam({"lr": 1e-8}), Trace_ELBO())
    guide(dataset)  # initialize
    print("Parameter shapes:")
    for name, param in guide.named_parameters():
        print(f"  {name}: {tuple(param.shape)}")

    for step in range(num_steps):
        svi.step(dataset)


if __name__ == "__main__":
    test_profile(backend="funsor", n=10, num_steps=100)
