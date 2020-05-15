# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.ops.arrowhead import SymmArrowhead, sqrt, triu_gram, triu_inverse, triu_matvecmul

from tests.common import assert_close


@pytest.mark.parametrize('head_size', [0, 2, 5])
def test_utilities(head_size):
    size = 5
    cov = torch.randn(size, size)
    cov = torch.mm(cov, cov.t())

    mask = torch.ones(size, size)
    mask[head_size:, head_size:] = 0.
    mask.view(-1)[::size + 1][head_size:] = 1.
    arrowhead_full = mask * cov
    expected = torch.flip(torch.flip(arrowhead_full, (-2, -1)).cholesky(), (-2, -1))
    # test if those flip ops give expected upper triangular values
    assert_close(expected.triu(), expected)
    assert_close(expected.matmul(expected.t()), arrowhead_full)

    # test sqrt
    arrowhead = SymmArrowhead(cov[:head_size], cov.diag()[head_size:])
    actual = sqrt(arrowhead)
    assert_close(actual.top, expected[:head_size])
    assert_close(actual.bottom_diag, expected.diag()[head_size:])

    # test triu_inverse
    expected = expected.inverse()
    actual = triu_inverse(actual)
    assert_close(actual.top, expected[:head_size])
    assert_close(actual.bottom_diag, expected.diag()[head_size:])

    # test triu_matvecmul
    v = torch.randn(size)
    assert_close(triu_matvecmul(actual, v), expected.matmul(v))
    assert_close(triu_matvecmul(actual, v, transpose=True),
                 expected.t().matmul(v))

    # test triu_gram
    actual = triu_gram(actual)
    expected = arrowhead_full.inverse() if head_size > 0 else arrowhead_full.diag().reciprocal()
    assert_close(actual, expected)
