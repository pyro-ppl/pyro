# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch

try:
    # This works in PyTorch 1.7+
    from torch.fft import irfft, rfft
except ModuleNotFoundError:
    # This works in PyTorch 1.6
    def rfft(input, n=None):
        if n is not None:
            m = input.size(-1)
            if n > m:
                input = torch.nn.functional.pad(input, (0, n - m))
            elif n < m:
                input = input[..., :n]
        return torch.view_as_complex(torch.rfft(input, 1))

    def irfft(input, n=None):
        if torch.is_complex(input):
            input = torch.view_as_real(input)
        else:
            input = torch.nn.functional.pad(input[..., None], (0, 1))
        if n is None:
            n = 2 * (input.size(-1) - 1)
        return torch.irfft(input, 1, signal_sizes=(n,))
