# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.ops.provenance import ProvenanceTensor, get_provenance, track_provenance
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
