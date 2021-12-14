# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

import pyro
import pyro.poutine as poutine
from pyro.poutine.runtime import get_mask, get_plates
from tests.common import assert_equal


def test_get_mask():
    assert get_mask() is None

    with poutine.mask(mask=True):
        assert get_mask() is True
    with poutine.mask(mask=False):
        assert get_mask() is False

    with pyro.plate("i", 2, dim=-1):
        mask1 = torch.tensor([False, True, True])
        mask2 = torch.tensor([True, True, False])
        with poutine.mask(mask=mask1):
            assert_equal(get_mask(), mask1)
            with poutine.mask(mask=mask2):
                assert_equal(get_mask(), mask1 & mask2)


def test_get_plates():
    def get_plate_names():
        plates = get_plates()
        assert isinstance(plates, tuple)
        return {f.name for f in plates}

    assert get_plate_names() == set()
    with pyro.plate("foo", 5):
        assert get_plate_names() == {"foo"}
        with pyro.plate("bar", 3):
            assert get_plate_names() == {"foo", "bar"}
