from __future__ import absolute_import, division, print_function

import math

import torch

# import opt_einsum as oe


def sumproduct(factors, target_dims, optimize=False):
    """
    """
    assert all(d < 0 for d in target_dims)

    if not optimize:
        product = factors[0]
        for factor in factors[1:]:
            product = product * factor

        for dim, size in enumerate(product.shape):
            if dim not in target_dims:
                product = product.sum(dim, True)

        target_ndims = 1-min(target_dims)
        while len(product.shape) > target_ndims:
            product = product.squeeze(0)
        return product

    raise NotImplementedError


def sumlogsumexp(log_factors, target_shape, optimize=False):
    """
    """
    # Handle trivial cases.
    if not any(isinstance(t, torch.Tensor) for t in log_factors):
        return math.exp(sum(log_factors))

    # Naive algorithm.
    if not optimize:
        log_result = log_factors[0]
        for log_factor in log_factors[1:]:
            log_result = log_result + log_factor
        result = log_result.exp()

        while result.dim() > len(target_shape):
            result = result.sum(0)
        while result.dim() < len(target_shape):
            result = result.unsqueeze(0)
        for dim, (result_size, target_size) in enumerate(zip(result.shape, target_shape)):
            if result_size > target_size:
                result = result.sum(dim, True)

        return result

    raise NotImplementedError
