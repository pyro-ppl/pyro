# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
import torch
import pyro.distributions as dist


def _hash(value):
    return tuple(value.tolist())


@pytest.mark.parametrize("num_destins", [1, 2, 3, 4, 5])
def test_enumerate(num_destins):
    num_sources = 2 * num_destins
    logits = torch.randn(num_sources, num_destins)
    d = dist.OneTwoMatching(logits)
    values = d.enumerate_support()
    logging.info("destins = {}, suport size = {}".format(num_destins, len(values)))
    assert d.support.check(values), "invalid"
    assert len(set(map(_hash, values))) == len(values), "not unique"
