# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

try:
    # This works in PyTorch 1.7+
    import torch.fft as torch_fft
except ModuleNotFoundError:
    # This works in PyTorch 1.6
    torch_fft = torch
