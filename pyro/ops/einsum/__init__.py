# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import opt_einsum

from pyro.util import ignore_jit_warnings

_PATH_CACHE = {}


def contract_expression(equation, *shapes, **kwargs):
    """
    Wrapper around :func:`opt_einsum.contract_expression` that optionally uses
    Pyro's cheap optimizer and optionally caches contraction paths.

    :param bool cache_path: whether to cache the contraction path.
        Defaults to True.
    """
    # memoize the contraction path
    cache_path = kwargs.pop('cache_path', True)
    if cache_path:
        kwargs_key = tuple(kwargs.items())
        key = equation, shapes, kwargs_key
        if key in _PATH_CACHE:
            return _PATH_CACHE[key]

    expr = opt_einsum.contract_expression(equation, *shapes, **kwargs)
    if cache_path:
        _PATH_CACHE[key] = expr
    return expr


def contract(equation, *operands, **kwargs):
    """
    Wrapper around :func:`opt_einsum.contract` that optionally uses Pyro's
    cheap optimizer and optionally caches contraction paths.

    :param bool cache_path: whether to cache the contraction path.
        Defaults to True.
    """
    backend = kwargs.pop('backend', 'numpy')
    out = kwargs.pop('out', None)
    shapes = [tuple(t.shape) for t in operands]
    with ignore_jit_warnings():
        expr = contract_expression(equation, *shapes)
        return expr(*operands, backend=backend, out=out)


__all__ = ['contract', 'contract_expression']
