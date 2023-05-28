# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.ops.provenance import ProvenanceTensor
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
