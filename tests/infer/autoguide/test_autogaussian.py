# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoGaussian
from pyro.infer.reparam import LocScaleReparam
from pyro.optim import Adam

BACKENDS = [
    "dense",
    pytest.param("funsor", marks=[pytest.mark.stage("funsor")]),
]


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


@pytest.mark.parametrize(
    "model", [pyrocov_model, pyrocov_model_relaxed, pyrocov_model_plated]
)
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


@pytest.mark.parametrize(
    "model", [pyrocov_model, pyrocov_model_relaxed, pyrocov_model_plated]
)
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


@pytest.mark.parametrize("backend", BACKENDS)
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
def test_profile(n=1, num_steps=1, backend="funsor"):
    """
    Helper function for profiling.
    """
    model = pyrocov_model_plated
    T, P, S, F = 2 * n, 3 * n, 4 * n, 5 * n
    dataset = {
        "features": torch.randn(S, F),
        "local_time": torch.randn(T, P),
        "weekly_strains": torch.randn(T, P, S).exp().round(),
    }

    guide = AutoGaussian(model)
    svi = SVI(model, guide, Adam({"lr": 1e-8}), Trace_ELBO())
    guide(dataset)  # initialize
    print("Factor inputs:")
    for name, inputs in guide._funsor_factor_inputs.items():
        print(f"  {name}:")
        for k, v in inputs.items():
            print(f"    {k}: {v}")
    print("Parameter shapes:")
    for name, param in guide.named_parameters():
        print(f"  {name}: {tuple(param.shape)}")

    for step in range(num_steps):
        svi.step(dataset)


if __name__ == "__main__":
    test_profile(n=10, num_steps=100, backend="funsor")
