from __future__ import absolute_import, division, print_function

import opt_einsum

from pyro.ops.einsum.shared import handle_sharing, shared_intermediates

_PATH_CACHE = {}


def contract(equation, *operands, **kwargs):
    """
    Like :func:`opt_einsum.contract` but works with
    :func:`~pyro.ops.einsum.shared_intermediates` contexts.

    :param bool cache_path: whether to cache the contraction path.
        Defaults to True.
    """
    backend = kwargs.pop('backend', 'numpy')
    with handle_sharing(backend) as backend:

        cache_path = kwargs.pop('cache_path', True)
        if not cache_path:
            return opt_einsum.contract(equation, *operands, backend=backend, **kwargs)

        # memoize the contraction path
        out = kwargs.pop('out', None)
        kwargs_key = tuple(kwargs.items())
        shapes = tuple(tuple(t.shape) for t in operands)
        key = equation, shapes, kwargs_key
        if key in _PATH_CACHE:
            expr = _PATH_CACHE[key]
        else:
            expr = opt_einsum.contract_expression(equation, *shapes, **kwargs)
            _PATH_CACHE[key] = expr
        return expr(*operands, backend=backend, out=out)


__all__ = ['contract', 'shared_intermediates']
