from __future__ import absolute_import, division, print_function

import contextlib
import os

import opt_einsum
from six.moves import cPickle as pickle

_PATH_CACHE = {}


def contract(equation, *operands, **kwargs):
    """
    Like :func:`opt_einsum.contract` but works with
    :func:`~opt_einsum.shared_intermediates` contexts.

    :param bool cache_path: whether to cache the contraction path.
        Defaults to True.
    """
    backend = kwargs.pop('backend', 'numpy')
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


@contextlib.contextmanager
def cached_paths(filename):
    """
    Context manager to load and save cached paths. This is most useful outside
    of the training loop, so that optimized einsum paths can be saved across
    training runs. Note that this saves even after user interrupt with CTRL-C.

    :param str filename: path to a pickle file where the cache will be stored
    """
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            _PATH_CACHE.update(pickle.load(f))
    try:
        yield
    finally:
        with open(filename, 'wb') as f:
            pickle.dump(_PATH_CACHE, f, pickle.HIGHEST_PROTOCOL)


__all__ = ['contract', 'cached_paths']
