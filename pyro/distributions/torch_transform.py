from __future__ import absolute_import, division, print_function

from pyro.distributions import ConditionalTransform
import torch
from torch.distributions.utils import _sum_rightmost


class TransformModule(torch.distributions.Transform, torch.nn.Module):
    """
    Transforms with learnable parameters such as normalizing flows should inherit from this class rather
    than `Transform` so they are also a subclass of `nn.Module` and inherit all the useful methods of that class.

    """

    def __init__(self, *args, **kwargs):
        super(TransformModule, self).__init__(*args, **kwargs)

    def __hash__(self):
        return super(torch.nn.Module, self).__hash__()


class ConditionalTransformModule(ConditionalTransform, torch.nn.Module):
    """
    Conditional transforms with learnable parameters such as normalizing flows should inherit from this class rather
    than `Transform` so they are also a subclass of `nn.Module` and inherit all the useful methods of that class.

    """

    def __init__(self, *args, **kwargs):
        super(ConditionalTransformModule, self).__init__(*args, **kwargs)

    def __hash__(self):
        return super(torch.nn.Module, self).__hash__()


class ComposeConditionalTransform(torch.distributions.ComposeTransform, ConditionalTransform):
    """
    Composes multiple transforms or conditional transforms in a chain.
    The transforms being composed are responsible for caching.

    Args:
        parts (list of :class:`Transform` and/or :class:`ConditionalTransform`): A list of transforms to compose.
    """

    def __call__(self, x, obs):
        for part in self.parts:
            if isinstance(part, torch.distributions.Transform):
                x = part(x)
            else:
                x = part(x, obs)
        return x

    def log_abs_det_jacobian(self, x, y, obs):
        if not self.parts:
            return torch.zeros_like(x)
        result = 0
        for part in self.parts[:-1]:
            if isinstance(part, torch.distributions.Transform):
                y_tmp = part(x)
                result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y_tmp),
                                                 self.event_dim - part.event_dim)
            else:
                y_tmp = part(x, obs)
                result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y_tmp, obs),
                                                 self.event_dim - part.event_dim)

            x = y_tmp
        part = self.parts[-1]
        if isinstance(part, torch.distributions.Transform):
            result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y),
                                             self.event_dim - part.event_dim)
        else:
            result = result + _sum_rightmost(part.log_abs_det_jacobian(x, y, obs),
                                             self.event_dim - part.event_dim)
        return result

    def __repr__(self):
        fmt_string = self.__class__.__name__ + '(\n    '
        fmt_string += ',\n    '.join([p.__repr__() for p in self.parts])
        fmt_string += '\n)'
        return fmt_string
