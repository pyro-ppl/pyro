# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.reparam import AutoReparam, MinimalReparam
from pyro.optim import Adam


def trace_name_is_observed(model):
    trace = poutine.trace(model).get_trace()
    return [
        (name, site["is_observed"])
        for name, site in trace.nodes.items()
        if site["type"] == "sample" and type(site["fn"]).__name__ != "_Subsample"
    ]


def normal_model():
    zero = torch.zeros(2)
    a = pyro.sample("a", dist.Normal(0, 1))
    b = pyro.sample("b", dist.LogNormal(0, 1))
    c = pyro.sample("c", dist.Normal(a, b))
    d = pyro.sample("d", dist.LogNormal(a, b))
    e = pyro.sample("e", dist.Normal(zero, b).to_event(1))
    f = pyro.sample("f", dist.LogNormal(zero, b).to_event(1))
    g = pyro.sample("g", dist.Normal(0, 1), obs=a)
    h = pyro.sample("h", dist.LogNormal(0, 1), obs=b)
    with pyro.plate("plate", 5):
        i = pyro.sample("i", dist.Normal(a, b))
        j = pyro.sample("j", dist.LogNormal(a, b))
    return a, b, c, d, e, f, g, h, i, j


def test_normal_minimal():
    model = MinimalReparam()(normal_model)
    actual = trace_name_is_observed(model)
    expected = [
        ("a", False),
        ("b", False),
        ("c", False),
        ("d", False),
        ("e", False),
        ("f", False),
        ("g", True),
        ("h", True),
        ("i", False),
        ("j", False),
    ]
    assert actual == expected


@pytest.mark.parametrize("centered", [None, 0.0, 1.0])
def test_normal_auto(centered):
    strategy = AutoReparam(centered=centered)
    model = strategy(normal_model)
    actual = trace_name_is_observed(model)
    if centered == 1.0:  # i.e. no decentering
        expected = [
            ("a", False),
            ("b_base", False),
            ("b", True),
            ("c", False),
            ("d_base", False),
            ("d", True),
            ("e", False),
            ("f_base", False),
            ("f", True),
            ("g", True),
            ("h", True),
            ("i", False),
            ("j_base", False),
            ("j", True),
        ]
    else:
        expected = [
            ("a_decentered", False),
            ("a", True),
            ("b_base_decentered", False),
            ("b_base", True),
            ("b", True),
            ("c_decentered", False),
            ("c", True),
            ("d_base_decentered", False),
            ("d_base", True),
            ("d", True),
            ("e_decentered", False),
            ("e", True),
            ("f_base_decentered", False),
            ("f_base", True),
            ("f", True),
            ("g", True),
            ("h", True),
            ("i_decentered", False),
            ("i", True),
            ("j_base_decentered", False),
            ("j_base", True),
            ("j", True),
        ]
    assert actual == expected

    # Also check that the config dict has been constructed.
    config = strategy.config
    assert isinstance(config, dict)
    model = poutine.reparam(normal_model, config)
    actual = trace_name_is_observed(model)
    assert actual == expected


def stable_model():
    zero = torch.zeros(2)
    a = pyro.sample("a", dist.Normal(0, 1))
    b = pyro.sample("b", dist.LogNormal(0, 1))
    c = pyro.sample("c", dist.Stable(1.5, 0.0, b, a))
    d = pyro.sample("d", dist.Stable(1.5, 0.0, b, 0.0), obs=a)
    e = pyro.sample("e", dist.Stable(1.5, 0.1, b, a))
    f = pyro.sample("f", dist.Stable(1.5, 0.1, b, 0.0), obs=a)
    g = pyro.sample("g", dist.Stable(1.5, zero, b, a).to_event(1))
    h = pyro.sample("h", dist.Stable(1.5, zero, b, 0).to_event(1), obs=a)
    i = pyro.sample(
        "i",
        dist.TransformedDistribution(
            dist.Stable(1.5, 0, b, a), dist.transforms.ExpTransform()
        ),
    )
    j = pyro.sample(
        "j",
        dist.TransformedDistribution(
            dist.Stable(1.5, 0, b, a), dist.transforms.ExpTransform()
        ),
        obs=a.exp(),
    )
    k = pyro.sample(
        "k",
        dist.TransformedDistribution(
            dist.Stable(1.5, zero, b, a), dist.transforms.ExpTransform()
        ).to_event(1),
    )
    l = pyro.sample(
        "l",
        dist.TransformedDistribution(
            dist.Stable(1.5, zero, b, a), dist.transforms.ExpTransform()
        ).to_event(1),
        obs=a.exp() + zero,
    )
    return a, b, c, d, e, f, g, h, i, j, k, l


def test_stable_minimal():
    model = MinimalReparam()(stable_model)
    actual = trace_name_is_observed(model)
    expected = [
        ("a", False),
        ("b", False),
        ("c_uniform", False),
        ("c_exponential", False),
        ("c", True),
        ("d_uniform", False),
        ("d_exponential", False),
        ("d", True),
        ("e_uniform", False),
        ("e_exponential", False),
        ("e", True),
        ("f_z_uniform", False),
        ("f_z_exponential", False),
        ("f_t_uniform", False),
        ("f_t_exponential", False),
        ("f", True),
        ("g_uniform", False),
        ("g_exponential", False),
        ("g", True),
        ("h_uniform", False),
        ("h_exponential", False),
        ("h", True),
        ("i_base_uniform", False),
        ("i_base_exponential", False),
        ("i_base", True),
        ("i", True),
        ("j_base_uniform", False),
        ("j_base_exponential", False),
        ("j_base", True),
        ("j", True),
        ("k_base_uniform", False),
        ("k_base_exponential", False),
        ("k_base", True),
        ("k", True),
        ("l_base_uniform", False),
        ("l_base_exponential", False),
        ("l_base", True),
        ("l", True),
    ]
    assert actual == expected


def test_stable_auto():
    model = AutoReparam()(stable_model)
    actual = trace_name_is_observed(model)
    expected = [
        ("a_decentered", False),
        ("a", True),
        ("b_base_decentered", False),
        ("b_base", True),
        ("b", True),
        ("c_decentered_uniform", False),
        ("c_decentered_exponential", False),
        ("c_decentered", True),
        ("c", True),
        ("d_uniform", False),
        ("d_exponential", False),
        ("d", True),
        ("e_decentered_uniform", False),
        ("e_decentered_exponential", False),
        ("e_decentered", True),
        ("e", True),
        ("f_z_uniform", False),
        ("f_z_exponential", False),
        ("f_t_uniform", False),
        ("f_t_exponential", False),
        ("f", True),
        ("g_decentered_uniform", False),
        ("g_decentered_exponential", False),
        ("g_decentered", True),
        ("g", True),
        ("h_uniform", False),
        ("h_exponential", False),
        ("h", True),
        ("i_base_decentered_uniform", False),
        ("i_base_decentered_exponential", False),
        ("i_base_decentered", True),
        ("i_base", True),
        ("i", True),
        ("j_base_uniform", False),
        ("j_base_exponential", False),
        ("j_base", True),
        ("j", True),
        ("k_base_decentered_uniform", False),
        ("k_base_decentered_exponential", False),
        ("k_base_decentered", True),
        ("k_base", True),
        ("k", True),
        ("l_base_uniform", False),
        ("l_base_exponential", False),
        ("l_base", True),
        ("l", True),
    ]
    assert actual == expected


def projected_normal_model():
    a = pyro.sample("a", dist.MultivariateNormal(torch.zeros(3), torch.eye(3)))
    b = pyro.sample("b", dist.ProjectedNormal(a))
    c = pyro.sample("c", dist.ProjectedNormal(torch.ones(3)))
    return a, b, c


def test_projected_normal_minimal():
    model = MinimalReparam()(projected_normal_model)
    actual = trace_name_is_observed(model)
    expected = [
        ("a", False),
        ("b_normal", False),
        ("b", True),
        ("c_normal", False),
        ("c", True),
    ]
    assert actual == expected


def test_projected_normal_auto():
    strategy = AutoReparam()
    model = strategy(projected_normal_model)
    actual = trace_name_is_observed(model)
    expected = [
        ("a", False),
        ("b_normal_decentered", False),
        ("b_normal", True),
        ("b", True),
        ("c_normal_decentered", False),
        ("c_normal", True),
        ("c", True),
    ]
    assert actual == expected


def softmax_model():
    a = pyro.sample("a", dist.Dirichlet(torch.ones(6)))
    b = pyro.sample("b", dist.RelaxedOneHotCategorical(probs=a, temperature=2.0))
    c = pyro.sample("c", dist.Normal(torch.ones(7), 1).to_event(1))
    d = pyro.sample("d", dist.RelaxedOneHotCategorical(logits=c, temperature=1.0))
    e = pyro.sample(
        "e", dist.RelaxedOneHotCategorical(logits=c, temperature=0.5), obs=d.round()
    )
    return a, b, c, d, e


def test_softmax_minimal():
    model = MinimalReparam()(softmax_model)
    actual = trace_name_is_observed(model)
    expected = [("a", False), ("b", False), ("c", False), ("d", False), ("e", True)]
    assert actual == expected


def test_softmax_auto():
    strategy = AutoReparam()
    model = strategy(softmax_model)
    actual = trace_name_is_observed(model)
    expected = [
        ("a", False),
        ("b_uniform", False),
        ("b", True),
        ("c_decentered", False),
        ("c", True),
        ("d_uniform", False),
        ("d", True),
        ("e", True),
    ]
    assert actual == expected


@pytest.mark.filterwarnings(
    "ignore:.*falling back to default initialization.*:RuntimeWarning"
)
@pytest.mark.parametrize("model", [normal_model, stable_model, projected_normal_model])
def test_end_to_end(model):
    # Test training.
    model = AutoReparam()(model)
    guide = AutoNormal(model)
    svi = SVI(model, guide, Adam({"lr": 1e-9}), Trace_ELBO())
    for step in range(3):
        svi.step()

    # Test prediction.
    predictive = Predictive(model, guide=guide, num_samples=2)
    samples = predictive()
    assert set("abc").issubset(samples.keys())
