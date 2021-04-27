from math import pi

import torch
from torch.distributions import Uniform

from pyro.distributions import constraints

from .torch_distribution import TorchDistribution


class SineSkewed(TorchDistribution):
    """ Distribution for breaking pointwise-symmetry on distributions over the d-dimensional torus.

    ** References: **
      1. Sine-skewed toroidal distributions and their application in protein bioinformatics
         Ameijeiras-Alonso, J., Ley, C. (2019)

    :param base_density: base density on the d-dimensional torus; event_shape must be [..., 2] where
        ``prod(event_shape[:-1]) == d``.
    :param skewness: skewness of the distribution; must have same shape as base_density.event_shape, all values
        must be in [-1,1] and ``abs(skewness).sum() <= 1``.
    """
    arg_constraints = {'skewness': constraints.interval(-1., 1.)}
    support = constraints.independent(constraints.real, 1)

    def __init__(self, base_density: TorchDistribution, skewness, validate_args=None):
        assert torch.all(skewness.abs() <= 1)
        assert torch.Size(base_density.event_shape) == skewness.shape
        assert base_density.event_shape[-1] == 2
        self.base_density = base_density
        self.skewness = skewness
        super().__init__(base_density.batch_shape, base_density.event_shape, validate_args=validate_args)

    def __repr__(self):
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]

        args_string = ', '.join(['{}: {}'.format(p, self.__dict__[p]
                                if self.__dict__[p].numel() == 1
                                else self.__dict__[p].size()) for p in param_names])
        return self.__class__.__name__ + '(' + f'base_density: {self.base_density.__repr__()}, ' + args_string + ')'

    def sample(self, sample_shape=torch.Size()):
        bd = self.base_density
        ys = bd.sample(sample_shape)
        mask = Uniform(0, 1.).sample(sample_shape) < 1. + (self.skewness * torch.sin((ys - bd.mean) % (2 * pi))).sum(-1)
        return torch.where(mask.view(*sample_shape, *(1 for _ in bd.event_shape)), ys, -ys + 2 * bd.mean)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        bd = self.base_density
        return bd.log_prob(value) + torch.log(1 + (self.skewness * torch.sin((value - bd.mean) % (2 * pi))).sum(-1))

    @classmethod
    def infer_shapes(cls, **arg_shapes):
        return arg_shapes['base_density'], arg_shapes['skewness']
