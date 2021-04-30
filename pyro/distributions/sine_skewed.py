from math import pi

import torch
from torch.distributions import Uniform

from pyro.distributions import constraints

from .torch_distribution import TorchDistribution


class SineSkewed(TorchDistribution):
    """The Sine Skewed distribution is a distribution for breaking pointwise-symmetry on a base-distributions over
    the d-dimensional torus.

    This distribution requires a base distribution on the torus. The parameter skewness can be inferred using
    :class:`~pyro.infer.HMC` or :class:`~pyro.infer.NUTS`. The following will produce a uniform prior
    over skewness,::

        def model(data, batch_shape, event_shape):
            n = torch.prod(torch.tensor(event_shape), -1, dtype=torch.int)
            skewness = torch.empty((*batch_shape, n)).view(-1, n)
            tots = torch.zeros(batch_shape).view(-1)
            for i in range(n):
                skewness[..., i] = pyro.sample(f'skew{i}', Uniform(0., 1 - tots))
                tots += skewness[..., i]
            sign = pyro.sample('sign', Uniform(0., torch.ones(skewness.shape)).to_event(len(skewness.shape)))
            skewness = torch.where(sign < .5, -skewness, skewness)

            if (*batch_shape, *event_shape) == tuple():
                skewness = skewness.reshape((*batch_shape, *event_shape))
            else:
                skewness = skewness.view(*batch_shape, *event_shape)

    .. note:: The base-distribution must be over a arbitrary dim torus.

    .. note:: ``skewness.abs().sum() <= 1.`` and ``(skewness.abs() <= 1).all()``.

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
        assert base_density.event_shape[-1] == 2
        assert skewness.shape[-1] == 2
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
        u = Uniform(0, 1.).sample(sample_shape + self.batch_shape)
        mask = u < 1. + (self.skewness * torch.sin((ys - bd.mean) % (2 * pi))).sum(-1)
        mask = mask.view(*sample_shape, *self.batch_shape, *(1 for _ in bd.event_shape))
        samples = (torch.where(mask, ys, -ys + 2 * bd.mean) + pi) % (2 * pi) - pi
        return samples

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        bd = self.base_density
        return bd.log_prob(value) + torch.log(1 + (self.skewness * torch.sin((value - bd.mean) % (2 * pi))).sum(-1))

    @classmethod
    def infer_shapes(cls, **arg_shapes):
        return arg_shapes['base_density'], arg_shapes['skewness']
