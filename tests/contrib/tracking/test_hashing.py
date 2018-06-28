from __future__ import absolute_import, division, print_function

import pytest
import torch
from torch.autograd import grad

import pyro
import pyro.distributions as dist
from pyro.contrib.tracking.hashing import LSH
from tests.common import assert_equal


@pytest.mark.parametrize('scale', [-1., 0., -1 * torch.ones(2, 2)])
def test_lsh_init(scale):
    num_error = 0
    try:
        lsh = LSH(scale)
    except AssertionError as e:
        num_error += 1
    else:
        pass
    finally:
        assert num_error > 0
