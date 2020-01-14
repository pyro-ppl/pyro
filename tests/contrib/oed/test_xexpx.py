# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.contrib.oed.eig import xexpx


@pytest.mark.parametrize("argument,output", [
    (torch.tensor([float('-inf')]), torch.tensor([0.])),
    (torch.tensor([0.]), torch.tensor([0.])),
    (torch.tensor([1.]), torch.exp(torch.tensor([1.])))
])
def test_xexpx(argument, output):
    assert xexpx(argument) == output
