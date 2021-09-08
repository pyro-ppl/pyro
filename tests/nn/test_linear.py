# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.nn.linear import FlatBatchLinear, FlatRank2Linear


def check_num_parameters(module):
    expected = 1
    for shape, batch_dims in [
        (module.shape_in, module.batch_dims_in),
        (module.shape_out, module.batch_dims_out),
    ]:
        for i, size in enumerate(shape):
            if i not in batch_dims:
                expected *= size

    actual = sum(p.numel() for p in module.parameters())
    assert actual == expected


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_dims_in", [{}, {0}, {1}, {0, 1}], ids=str)
@pytest.mark.parametrize("cls", [FlatBatchLinear, FlatRank2Linear])
def test_1(cls, batch_dims_in, sample_shape):
    shape_in = torch.Size((4, 3, 2))
    shape_out = torch.Size((5, 4, 3))
    batch_dims_out = {i + 1 for i in batch_dims_in}

    f = cls(shape_in, shape_out, batch_dims_in, batch_dims_out)
    if cls == FlatBatchLinear:
        check_num_parameters(f)

    x = torch.randn(sample_shape + (shape_in.numel(),))
    y = f(x)
    assert y.shape == sample_shape + (shape_out.numel(),)


@pytest.mark.parametrize("sample_shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("batch_dims_in", [{}, {2}, {4}, {2, 4}], ids=str)
@pytest.mark.parametrize("cls", [FlatBatchLinear, FlatRank2Linear])
def test_2(cls, batch_dims_in, sample_shape):
    shape_in = torch.Size((5, 1, 3, 1, 1))
    shape_out = torch.Size((4, 3, 2, 1))
    batch_dims_out = {i - 1 for i in batch_dims_in}

    f = cls(shape_in, shape_out, batch_dims_in, batch_dims_out)
    if cls == FlatBatchLinear:
        check_num_parameters(f)

    x = torch.randn(sample_shape + (shape_in.numel(),))
    y = f(x)
    assert y.shape == sample_shape + (shape_out.numel(),)
