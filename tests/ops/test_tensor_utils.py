import pytest
from tests.common import assert_equal

import torch
from pyro.ops.tensor_utils import block_diag, conv1d_fft


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


@pytest.mark.parametrize('T', [2, 3, 4, 5, 10])
@pytest.mark.parametrize('batch_shape', [(), (4,), (2, 3)], ids=str)
def test_conv1d_fft(batch_shape, T):
    signal = torch.randn(T)  # not batched
    kernel = torch.randn(*batch_shape, T).exp()  # batched
    actual = conv1d_fft(signal, kernel)

    # FIXME is this correct?
    padded = torch.cat([torch.zeros_like(signal), signal])
    expected = torch.stack([kernel @ padded[t:t+T]
                            for t in range(1, 1+T)], dim=-1)
    assert actual.shape == expected.shape
    assert torch.allclose(actual, expected), (actual, expected)
