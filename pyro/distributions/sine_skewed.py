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

    .. note:: An event in the base-distribution must be on a d-torus so the event_shape must be (d,2) or (2,).

    .. note:: For the skewness parameter it must hold that the sum of the absolute value of its weights for an event
        must be less than or equal to one. See eq. 2.1 in [1].

    ** References: **
      1. Sine-skewed toroidal distributions and their application in protein bioinformatics
         Ameijeiras-Alonso, J., Ley, C. (2019)

    :param base_density: base density on the d-dimensional torus.
    :param skewness: skewness of the distribution.
    """
    arg_constraints = {'skewness': constraints.interval(-1., 1.)}
    support = constraints.independent(constraints.real, 1)

    def __init__(self, base_density: TorchDistribution, skewness, validate_args=None):
        assert base_density.event_shape[-1] == 2 and len(base_density.event_shape) <= 2
        assert base_density.shape()[len(base_density.shape()) - len(skewness.shape):] == skewness.shape
        assert (skewness.abs().sum(-1 if len(skewness.shape) == 1 else (-2, -1)) <= 1).all()

        self.base_density = base_density
        self.skewness = skewness.broadcast_to(base_density.shape())
        super().__init__(base_density.batch_shape, base_density.event_shape, validate_args=validate_args)

        if self._validate_args and base_density.mean.device != skewness.device:
            raise ValueError(f"base_density: {base_density.__class__.__name__} and {self.__class__.__name__} "
                             f"must be on same device.")

    def __repr__(self):
        param_names = [k for k, _ in self.arg_constraints.items() if k in self.__dict__]

        args_string = ', '.join(['{}: {}'.format(p, self.__dict__[p]
        if self.__dict__[p].numel() == 1
        else self.__dict__[p].size()) for p in param_names])
        return self.__class__.__name__ + '(' + f'base_density: {self.base_density.__repr__()}, ' + args_string + ')'

    def sample(self, sample_shape=torch.Size()):
        bd = self.base_density
        ys = bd.sample(sample_shape)
        u = Uniform(0., torch.ones(torch.Size([]), device=self.skewness.device)).sample(sample_shape + self.batch_shape)

        mask = u < 1. + (self.skewness * torch.sin((ys - bd.mean) % (2 * pi))).view(*(sample_shape + self.batch_shape),
                                                                                    -1).sum(-1)
        mask = mask.view(*sample_shape, *self.batch_shape, *(1 for _ in bd.event_shape))
        samples = (torch.where(mask, ys, -ys + 2 * bd.mean) + pi) % (2 * pi) - pi
        return samples

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        flat_event = torch.tensor(self.event_shape, device=value.device).prod()
        bd = self.base_density
        bd_prob = bd.log_prob(value)
        sine_prob = torch.log(
            1 + (self.skewness * torch.sin((value - bd.mean) % (2 * pi))).reshape((-1, flat_event)).sum(-1))
        return (bd_prob.view((-1)) + sine_prob).view(bd_prob.shape)

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        new = self._get_checked_instance(SineSkewed, _instance)
        base_dist = self.base_density.expand(batch_shape, None)
        new.base_density = base_dist
        for name in self.arg_constraints:
            setattr(new, name, getattr(self, name).expand((*batch_shape, *self.event_shape)))
        super(SineSkewed, new).__init__(batch_shape, self.event_shape, validate_args=None)
        new._validate_args = self._validate_args
        return new
