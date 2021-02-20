# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from pyro.contrib.epidemiology.util import compute_bin_probs


@pytest.mark.parametrize("num_quant_bins", [2, 4, 8, 12, 16])
def test_quantization_scheme(num_quant_bins, num_samples=1000 * 1000):
    min, max = 0, 7
    probs = torch.zeros(max + 1)

    x = torch.linspace(-0.5, max + 0.5, num_samples)
    bin_probs = compute_bin_probs(x - x.floor(), num_quant_bins=num_quant_bins)
    x_floor = x.floor()

    q_min = 1 - num_quant_bins // 2
    q_max = 1 + num_quant_bins // 2

    for k, q in enumerate(range(q_min, q_max)):
        y = (x_floor + q).long()
        y = torch.max(y, 2 * min - 1 - y)
        y = torch.min(y, 2 * max + 1 - y)
        probs.scatter_add_(0, y, bin_probs[:, k] / num_samples)

    max_deviation = (probs - 1.0 / (max + 1.0)).abs().max().item()
    assert max_deviation < 1.0e-4
