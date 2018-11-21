from __future__ import absolute_import, division, print_function


def require_backward(tensor):
    """
    Marks a tensor as an adjoint leaf.
    """
    tensor._pyro_backward_result = None


def requires_backward(tensor):
    """
    Returns true for internal and leaf nodes of the adjoint graph.
    """
    return (hasattr(tensor, '_pyro_backward_result') or
            hasattr(tensor, '_pyro_backward'))


class _TransposeBackward(object):
    def __init__(self, a, axes):
        self.a = a
        self.axes = axes

    def __call__(self, sample):
        inv_axes = [None] * len(self.axes)
        for i, j in enumerate(self.axes):
            inv_axes[j] = i
        self.a._pyro_backward_result = sample.permute(inv_axes)


# this requires https://github.com/dgasmith/opt_einsum/pull/74
def transpose(a, axes):
    result = a.permute(axes)
    if requires_backward(a):
        result._pyro_backward = _TransposeBackward(a, axes)
    return result
