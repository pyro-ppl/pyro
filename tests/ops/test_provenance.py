# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer.inspect import get_model_relations
from pyro.ops.provenance import (
    ProvenanceTensor,
    detach_provenance,
    extract_provenance,
    get_provenance,
    track_provenance,
)
from tests.common import assert_equal, requires_cuda


@requires_cuda
@pytest.mark.parametrize(
    "dtype1",
    [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ],
)
@pytest.mark.parametrize(
    "dtype2",
    [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ],
)
def test_provenance_tensor(dtype1, dtype2):
    device = torch.device("cuda")
    x = torch.tensor([1, 2, 3], dtype=dtype1)
    y = ProvenanceTensor(x, frozenset(["x"]))
    z = torch.as_tensor(y, device=device, dtype=dtype2)

    assert x.shape == y.shape == z.shape
    assert_equal(x, z.cpu())


@pytest.mark.parametrize(
    "x",
    [
        torch.tensor([1, 2, 3]),
        track_provenance(torch.tensor([1, 2, 3]), frozenset("y")),
        frozenset([torch.tensor([0, 1]), torch.tensor([2, 3])]),
        set([torch.tensor([0, 1]), torch.tensor([2, 3])]),
        [torch.tensor([0, 1]), torch.tensor([2, 3])],
        (torch.tensor([0, 1]), torch.tensor([2, 3])),
        {"a": torch.tensor([0, 1]), "b": torch.tensor([2, 3])},
        {
            "a": track_provenance(torch.tensor([0, 1]), frozenset("y")),
            "b": [torch.tensor([2, 3]), torch.tensor([4, 5])],
        },
    ],
)
def test_track_provenance(x):
    new_provenance = frozenset("x")
    old_provenance = get_provenance(x)
    provenance = old_provenance | new_provenance
    assert provenance == get_provenance(track_provenance(x, new_provenance))


@pytest.mark.parametrize(
    "x",
    [
        torch.Size([3]),
        torch.Size([2, 5]),
        torch.Size([]),
        [(torch.Size([3]),), {}],
        [(), {"shape": torch.Size([2, 5])}],
        [(torch.zeros(2), torch.Size([3])), {}],
    ],
    ids=["size", "size_2d", "size_empty", "in_args", "in_kwargs", "mixed_with_tensor"],
)
def test_provenance_preserves_torch_size(x):
    """torch.Size subclasses tuple, so it must not be rebuilt as a plain tuple."""

    def assert_sizes_intact(original, result):
        if isinstance(original, torch.Tensor):
            return  # tensors are legitimately wrapped/unwrapped
        assert type(original) is type(result)
        if isinstance(original, torch.Size):
            assert result == original
        elif isinstance(original, (list, tuple)):
            for a, b in zip(original, result):
                assert_sizes_intact(a, b)
        elif isinstance(original, dict):
            for key in original:
                assert_sizes_intact(original[key], result[key])

    assert_sizes_intact(x, track_provenance(x, frozenset({"a"})))
    assert_sizes_intact(x, extract_provenance(x)[0])
    assert_sizes_intact(x, detach_provenance(x))


def test_multinomial_render_model():
    """https://github.com/pyro-ppl/pyro/issues/3436"""

    def model():
        probs = pyro.param("probs", torch.tensor([0.1, 0.2, 0.7]))
        return pyro.sample("y", dist.Multinomial(total_count=10, probs=probs))

    relations = get_model_relations(model)
    assert "y" in relations["sample_sample"] or "y" in relations["sample_param"]


def test_provenance_tensor_new_with_size():
    """Tensor.new() reads a torch.Size as a shape but a tuple as data."""
    x = ProvenanceTensor(torch.zeros(4, dtype=torch.long), frozenset({"x"}))
    assert tuple(x.new(torch.Size([3])).shape) == (3,)
