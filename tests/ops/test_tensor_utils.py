# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
import scipy.fftpack as fftpack
import torch

import pyro
from pyro.ops.tensor_utils import (block_diag_embed, block_diagonal, convolve, dct, idct, next_fast_len,
                                   periodic_cumsum, periodic_features, periodic_repeat, precision_to_scale_tril,
                                   repeated_matmul)
from tests.common import assert_close, assert_equal

pytestmark = pytest.mark.stage('unit')


@pytest.mark.parametrize('batch_size', [1, 2, 3])
@pytest.mark.parametrize('block_size', [torch.Size([2, 2]), torch.Size([3, 1]), torch.Size([4, 2])])
def test_block_diag_embed(batch_size, block_size):
    m = torch.randn(block_size).unsqueeze(0).expand((batch_size,) + block_size)
    b = block_diag_embed(m)

    assert b.shape == (batch_size * block_size[0], batch_size * block_size[1])

    assert_equal(b.sum(), m.sum())

    for k in range(batch_size):
        bottom, top = k * block_size[0], (k + 1) * block_size[0]
        left, right = k * block_size[1], (k + 1) * block_size[1]
        assert_equal(b[bottom:top, left:right], m[k])


@pytest.mark.parametrize('batch_shape', [torch.Size([]), torch.Size([7])])
@pytest.mark.parametrize('mat_size,block_size', [(torch.Size([2, 2]), 2), (torch.Size([3, 1]), 1),
                                                 (torch.Size([6, 3]), 3)])
def test_block_diag(batch_shape, mat_size, block_size):
    mat = torch.randn(batch_shape + (block_size,) + mat_size)
    mat_embed = block_diag_embed(mat)
    mat_embed_diag = block_diagonal(mat_embed, block_size)
    assert_equal(mat_embed_diag, mat)


@pytest.mark.parametrize("size", [5, 6, 7, 8])
@pytest.mark.parametrize("period", [2, 3, 4])
@pytest.mark.parametrize("left_shape", [(), (6,), (3, 2)], ids=str)
@pytest.mark.parametrize("right_shape", [(), (7,), (5, 4)], ids=str)
def test_periodic_repeat(period, size, left_shape, right_shape):
    dim = -1 - len(right_shape)
    tensor = torch.randn(left_shape + (period,) + right_shape)
    actual = periodic_repeat(tensor, size, dim)
    assert actual.shape == left_shape + (size,) + right_shape
    dots = (slice(None),) * len(left_shape)
    for t in range(size):
        assert_equal(actual[dots + (t,)], tensor[dots + (t % period,)])


@pytest.mark.parametrize("duration", range(3, 100))
def test_periodic_features(duration):
    pyro.set_rng_seed(duration)
    max_period = torch.distributions.Uniform(2, duration).sample().item()
    for max_period in [max_period, duration]:
        min_period = torch.distributions.Uniform(2, max_period).sample().item()
        for min_period in [min_period, 2]:
            actual = periodic_features(duration, max_period, min_period)
            assert actual.shape == (duration, 2 * math.ceil(max_period / min_period) - 2)
            assert (-1 <= actual).all()
            assert (actual <= 1).all()


@pytest.mark.parametrize("size", [5, 6, 7, 8])
@pytest.mark.parametrize("period", [2, 3, 4])
@pytest.mark.parametrize("left_shape", [(), (6,), (3, 2)], ids=str)
@pytest.mark.parametrize("right_shape", [(), (7,), (5, 4)], ids=str)
def test_periodic_cumsum(period, size, left_shape, right_shape):
    dim = -1 - len(right_shape)
    tensor = torch.randn(left_shape + (size,) + right_shape)
    actual = periodic_cumsum(tensor, period, dim)
    assert actual.shape == tensor.shape
    dots = (slice(None),) * len(left_shape)
    for t in range(period):
        assert_equal(actual[dots + (t,)], tensor[dots + (t,)])
    for t in range(period, size):
        assert_close(actual[dots + (t,)], tensor[dots + (t,)] + actual[dots + (t - period,)])


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
@pytest.mark.parametrize('mode', ['full', 'valid', 'same'])
def test_convolve(batch_shape, m, n, mode):
    signal = torch.randn(*batch_shape, m)
    kernel = torch.randn(*batch_shape, n)
    actual = convolve(signal, kernel, mode)
    expected = torch.stack([
        torch.tensor(np.convolve(s, k, mode=mode))
        for s, k in zip(signal.reshape(-1, m), kernel.reshape(-1, n))
    ]).reshape(*batch_shape, -1)
    assert_close(actual, expected)


@pytest.mark.parametrize('size', [torch.Size([2, 2]), torch.Size([4, 3, 3]), torch.Size([4, 1, 2, 2])])
@pytest.mark.parametrize('n', [1, 2, 3, 7, 8])
def test_repeated_matmul(size, n):
    M = torch.randn(size)
    result = repeated_matmul(M, n)
    assert result.shape == ((n,) + size)

    serial_result = M
    for i in range(n):
        assert_equal(result[i, ...], serial_result)
        serial_result = torch.matmul(serial_result, M)


@pytest.mark.parametrize('shape', [(3, 4), (5,), (2, 1, 6)])
def test_dct(shape):
    x = torch.randn(shape)
    actual = dct(x)
    expected = torch.from_numpy(fftpack.dct(x.numpy(), norm='ortho'))
    assert_close(actual, expected)


@pytest.mark.parametrize('shape', [(3, 4), (5,), (2, 1, 6)])
def test_idct(shape):
    x = torch.randn(shape)
    actual = idct(x)
    expected = torch.from_numpy(fftpack.idct(x.numpy(), norm='ortho'))
    assert_close(actual, expected)


@pytest.mark.parametrize("dim", [-4, -3, -2, -1, 0, 1, 2, 3])
@pytest.mark.parametrize("fn", [dct, idct])
def test_dct_dim(fn, dim):
    x = torch.randn(4, 5, 6, 7)
    actual = fn(x, dim=dim)
    if dim == -1 or dim == 3:
        expected = fn(x)
    else:
        expected = fn(x.transpose(-1, dim)).transpose(-1, dim)
    assert_close(actual, expected)


def test_next_fast_len():
    for size in range(1, 1000):
        assert next_fast_len(size) == fftpack.next_fast_len(size)


@pytest.mark.parametrize('batch_shape,event_shape', [
    ((), (5,)),
    ((3,), (4,)),
])
def test_precision_to_scale_tril(batch_shape, event_shape):
    x = torch.randn(batch_shape + event_shape + event_shape)
    precision = x.matmul(x.transpose(-2, -1))
    actual = precision_to_scale_tril(precision)
    expected = precision.inverse().cholesky()
    assert_close(actual, expected)
