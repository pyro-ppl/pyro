# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch

from pyro.ops.indexing import Vindex

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor

    import pyro.contrib.funsor

    funsor.set_backend("torch")
    from pyroapi import distributions as dist
    from pyroapi import infer, pyro

    from tests.contrib.funsor.test_valid_models_enum import assert_ok
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("subsampling", [False, True])
@pytest.mark.parametrize("enumerate_", [None, "parallel", "sequential"])
def test_enum_discrete_iplate_plate_dependency_ok(subsampling, enumerate_):
    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={"enumerate": "parallel"})
        inner_plate = pyro.plate("plate", 10, subsample_size=4 if subsampling else None)
        for i in pyro.plate(
            "iplate", 10, subsample=torch.arange(3) if subsampling else None
        ):
            pyro.sample("y_{}".format(i), dist.Bernoulli(0.5))
            with inner_plate:
                pyro.sample(
                    "x_{}".format(i),
                    dist.Bernoulli(0.5),
                    infer={"enumerate": enumerate_},
                )

    assert_ok(model, max_plate_nesting=1)


def test_enum_iplate_iplate_ok():
    @infer.config_enumerate
    def model(data=None):
        probs_a = torch.tensor([0.45, 0.55])
        probs_b = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
        probs_c = torch.tensor([[0.75, 0.25], [0.55, 0.45]])
        probs_d = torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]])

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

    data = torch.tensor([[0, 1], [0, 0]])
    assert_ok(model, max_plate_nesting=1, data=data)


def test_enum_plate_iplate_ok():
    @infer.config_enumerate
    def model(data=None):
        probs_a = torch.tensor([0.45, 0.55])
        probs_b = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
        probs_c = torch.tensor([[0.75, 0.25], [0.55, 0.45]])
        probs_d = torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]])

        b_axis = pyro.plate("b_axis", 2)
        c_axis = pyro.plate("c_axis", 2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        with b_axis:
            for j in c_axis:
                c_j = pyro.sample("c_{}".format(j), dist.Categorical(probs_c[a]))
                pyro.sample(
                    "d_{}".format(j),
                    dist.Categorical(Vindex(probs_d)[b, c_j]),
                    obs=data[:, j],
                )

    data = torch.tensor([[0, 1], [0, 0]])
    assert_ok(model, max_plate_nesting=1, data=data)


def test_enum_iplate_plate_ok():
    @infer.config_enumerate
    def model(data=None):
        probs_a = torch.tensor([0.45, 0.55])
        probs_b = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
        probs_c = torch.tensor([[0.75, 0.25], [0.55, 0.45]])
        probs_d = torch.tensor([[[0.4, 0.6], [0.3, 0.7]], [[0.3, 0.7], [0.2, 0.8]]])

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

    data = torch.tensor([[0, 1], [0, 0]])
    assert_ok(model, max_plate_nesting=1, data=data)
