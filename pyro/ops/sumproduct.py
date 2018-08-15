from __future__ import absolute_import, division, print_function

import math

import torch

# import opt_einsum as oe


def sumproduct(factors, target_shape, optimize=False):
    # Handle trivial cases.
    if not any(isinstance(t, torch.Tensor) for t in factors):
        result = 1.
        for factor in factors:
            result *= factor
        return result

    # Naive algorithm.
    if not optimize:
        result = factors[0]
        for factor in factors[1:]:
            result = result * factor

        while result.dim() > len(target_shape):
            result = result.sum(0)
        while result.dim() < len(target_shape):
            result = result.unsqueeze(0)
        for dim, (result_size, target_size) in enumerate(zip(result.shape, target_shape)):
            if result_size > target_size:
                result = result.sum(dim, True)

        return result

    # Use opt-einsum.
    raise NotImplementedError
