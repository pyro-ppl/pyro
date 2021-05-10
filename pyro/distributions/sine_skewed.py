import warnings
from math import pi

import torch
from torch import broadcast_shapes
from torch.distributions import Uniform

from pyro.distributions import constraints

from .torch_distribution import TorchDistribution


class SineSkewed(TorchDistribution):
    """The Sine Skewed distribution [1] is a distribution for breaking pointwise-symmetry on a base-distribution over
    the d-dimensional torus defined as ⨂^d S^1 where S^1 is the circle. So for example the 0-torus is a point, the
    1-torus is a circle and the 2-tours is commonly associated with the donut shape (some may object to this simile).

    The skewness parameter can be inferred using :class:`~pyro.infer.HMC` or :class:`~pyro.infer.NUTS`.
    For example, the following will produce a uniform prior over skewness for the 2-torus,::

        def model(...):
            ...
            skewness_phi = pyro.sample(f'skewness_phi', Uniform(skewness.abs().sum(), 1 - tots))
            psi_bound = 1 - skewness_phi.abs()
            skewness_psi = pyro.sample(f'skewness_psi', Uniform(-psi_bound, psi_bound)
            skewness = torch.stack((skewness_phi, skewness_psi), dim=0)
            ...

    In the context of :class:`~pyro.infer.SVI`, this distribution can be freely used as a likelihood, but use as a
    latent variables will lead to slow inference for 2 and higher order toruses. This is because the base_dist
    cannot be reparameterized.

    .. note:: An event in the base distribution must be on a d-torus, so the event_shape must be (d,).

    .. note:: For the skewness parameter, it must hold that the sum of the absolute value of its weights for an event
        must be less than or equal to one. See eq. 2.1 in [1].

    ** References: **
      1. Sine-skewed toroidal distributions and their application in protein bioinformatics
         Ameijeiras-Alonso, J., Ley, C. (2019)

    :param base_dist: base density on a d-dimensional torus.
    :param skewness: skewness of the distribution.
    """
    arg_constraints = {'skewness': constraints.independent(constraints.interval(-1., 1.), 1)}

    support = constraints.independent(constraints.real, 1)

    def __init__(self, base_dist: TorchDistribution, skewness, validate_args=None):
        if (skewness.abs().sum(-1) > 1.).any():
            warnings.warn("Total skewness weight shouldn't exceed one.", UserWarning)

        batch_shape = broadcast_shapes(base_dist.batch_shape, skewness.shape[:-1])
        event_shape = skewness.shape[-1:]
        self.skewness = skewness.broadcast_to(batch_shape + event_shape)
        self.base_dist = base_dist.expand(batch_shape)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        if self._validate_args and base_dist.mean.device != skewness.device:
            raise ValueError(f"base_density: {base_dist.__class__.__name__} and SineSkewed "
                             f"must be on same device.")

    def __repr__(self):
        args_string = ', '.join(['{}: {}'.format(p, getattr(self, p)
                                if getattr(self, p).numel() == 1
                                else getattr(self, p).size()) for p in self.arg_constraints.keys()])
        return self.__class__.__name__ + '(' + f'base_density: {str(self.base_dist)}, ' + args_string + ')'

    def sample(self, sample_shape=torch.Size()):
        bd = self.base_dist
        ys = bd.sample(sample_shape)
        u = Uniform(0., torch.ones(torch.Size([]), device=self.skewness.device)).sample(sample_shape + self.batch_shape)

        # Section 2.3 step 3 in [1]
        mask = u < .5 + .5 * (self.skewness * torch.sin((ys - bd.mean) % (2 * pi))).sum(-1)
        mask = mask[..., None]
        samples = (torch.where(mask, ys, -ys + 2 * bd.mean) + pi) % (2 * pi) - pi
        return samples

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        # Eq. 2.1 in [1]
        skew_prob = torch.log(1 + (self.skewness * torch.sin((value - self.base_dist.mean) % (2 * pi))).sum(-1))
        return self.base_dist.log_prob(value) + skew_prob

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        new = self._get_checked_instance(SineSkewed, _instance)
        base_dist = self.base_dist.expand(batch_shape)
        new.base_dist = base_dist
        new.skewness = self.skewness.expand(batch_shape + (-1,))
        super(SineSkewed, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
