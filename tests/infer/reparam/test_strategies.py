# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer.reparam import AutoReparam, MinimalReparam


def trace_name_is_observed(model):
    trace = poutine.trace(model).get_trace()
    return [
        (name, site["is_observed"])
        for name, site in trace.nodes.items()
        if site["type"] == "sample"
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
    return a, b, c, d, e, f, g, h


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
    ]
    assert actual == expected


def test_normal_auto():
    model = AutoReparam()(normal_model)
    actual = trace_name_is_observed(model)
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
    ]
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
