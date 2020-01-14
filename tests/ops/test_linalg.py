# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.ops.linalg import rinverse
from tests.common import assert_equal


@pytest.mark.parametrize("A", [
    torch.tensor([[17.]]),
    torch.tensor([[1., 2.], [2., -3.]]),
    torch.tensor([[1., 2, 0], [2, -2, 4], [0, 4, 5]]),
    torch.tensor([[1., 2, 0, 7], [2, -2, 4, -1], [0, 4, 5, 8], [7, -1, 8, 1]]),
    torch.tensor([[1., 2, 0, 7, 0], [2, -2, 4, -1, 2], [0, 4, 5, 8, -4], [7, -1, 8, 1, -3], [0, 2, -4, -3, -1]]),
    torch.eye(40)
    ])
@pytest.mark.parametrize("use_sym", [True, False])
def test_sym_rinverse(A, use_sym):
    d = A.shape[-1]
    assert_equal(rinverse(A, sym=use_sym), torch.inverse(A), prec=1e-8)
    assert_equal(torch.mm(A, rinverse(A, sym=use_sym)), torch.eye(d), prec=1e-8)
    batched_A = A.unsqueeze(0).unsqueeze(0).expand(5, 4, d, d)
    expected_A = torch.inverse(A).unsqueeze(0).unsqueeze(0).expand(5, 4, d, d)
    assert_equal(rinverse(batched_A, sym=use_sym), expected_A, prec=1e-8)
