import pytest
import numpy as np
import torch

from pyro.ops.tensor_utils import block_diag, convolve
from tests.common import assert_equal, assert_close

pytestmark = pytest.mark.stage('unit')


@pytest.mark.parametrize('batch_size', [1, 2, 3])
@pytest.mark.parametrize('block_size', [torch.Size([2, 2]), torch.Size([3, 1]), torch.Size([4, 2])])
def test_block_diag(batch_size, block_size):
    m = torch.randn(block_size).unsqueeze(0).expand((batch_size,) + block_size)
    b = block_diag(m)

    assert b.shape == (batch_size * block_size[0], batch_size * block_size[1])

    assert_equal(b.sum(), m.sum())

    for k in range(batch_size):
        bottom, top = k * block_size[0], (k + 1) * block_size[0]
        left, right = k * block_size[1], (k + 1) * block_size[1]
        assert_equal(b[bottom:top, left:right], m[k])


@pytest.mark.parametrize('m', [2, 3, 4, 5, 6, 10])
@pytest.mark.parametrize('n', [2, 3, 4, 5, 6, 10])
@pytest.mark.parametrize('mode', ['full', 'valid', 'same'])
def test_convolve_shape(m, n, mode):
    signal = torch.randn(m)
    kernel = torch.randn(n)
    actual = convolve(signal, kernel, mode)
    expected = np.convolve(signal, kernel, mode=mode)
    assert actual.shape == expected.shape


@pytest.mark.parametrize('m', [2, 3, 4, 5, 6, 10])
@pytest.mark.parametrize('n', [2, 3, 4, 5, 6, 10])
@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 3)], ids=str)
@pytest.mark.parametrize('mode', ['full', 
    #'valid',
    'same'])
def test_convolve(batch_shape, m, n, mode):
    signal = torch.randn(*batch_shape, m)
    kernel = torch.randn(*batch_shape, n)
    actual = convolve(signal, kernel, mode)
    expected = torch.stack([
        torch.tensor(np.convolve(s, k, mode=mode))
        for s, k in zip(signal.reshape(-1, m), kernel.reshape(-1, n))
    ]).reshape(*batch_shape, -1)
    assert_close(actual, expected)
