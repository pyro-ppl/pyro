# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pytest
from pyro.distributions.transforms import HaarTransform
from tests.common import assert_equal


@pytest.mark.parametrize('size', [1, 3, 4, 7, 8, 9])
def test_haar_ortho(size):
    haar = HaarTransform()
    eye = torch.eye(size)
    mat = haar(eye)
    assert_equal(eye, mat @ mat.t())
