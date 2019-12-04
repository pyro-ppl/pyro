import torch
from torch.distributions import constraints
from torch.distributions.transforms import AffineTransform
from pyro.distributions.torch import TransformedDistribution, Cauchy
from pyro.distributions.torch_distribution import TorchDistributionMixin



class MultivariateCauchy(TransformedDistribution):
    r"""
    Creates a multivariate Cauchy distribution parameterized by
    :attr:`loc` and :attr:`scale_tril` where::

        X ~ Cauchy(loc, 1.0)
        Y = scale_tril @ X ~ MultivariateCauchy(loc, scale_tril)

    :param torch.tensor loc: the location parameter of the distribution, a D-dimensional vector.
    :param torch.tensor scale_tril: the scale parameter of the distribution, a D x D lower-triangular matrix.
    """
    arg_constraints = {'loc': constraints.real, 'scale_tril': constraints.lower_triangular}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale_tril, validate_args=None):
        base_dist = Cauchy(loc, 1.0).to_event(1)
        transforms = [AffineTransform(torch.zeros_like(loc), scale_tril, event_dim=1)]
        super(MultivariateCauchy, self).__init__(base_dist, transforms, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateCauchy, _instance)
        return super(MultivariateCauchy, self).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale_tril

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        raise ValueError("Distribution has infinite variance.")
