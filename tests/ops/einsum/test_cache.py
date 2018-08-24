from __future__ import absolute_import, division, print_function

import os
import pytest
import torch

from pyro.ops.einsum import cached_paths, contract
from tests.common import TemporaryDirectory


def perform_computation(num_steps):
    x = torch.randn(2, 3)
    y = torch.randn(3, 4)
    z = torch.randn(4, 5)
    for i in range(num_steps):
        contract("ab,bc,cd->ad", x, y, z)


@pytest.mark.parametrize('num_steps', [1, 2, 3])
def test_cached_paths(num_steps):
    with TemporaryDirectory() as dirname:
        filename = os.path.join(dirname, 'paths.pkl')
        assert not os.path.exists(filename)
        with cached_paths(filename):
            assert not os.path.exists(filename)
            perform_computation(num_steps)
            assert not os.path.exists(filename)
        assert os.path.exists(filename)
        with cached_paths(filename):
            assert os.path.exists(filename)
            perform_computation(num_steps)
            assert os.path.exists(filename)
        assert os.path.exists(filename)
    assert not os.path.exists(filename)
