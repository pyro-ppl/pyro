import torch
from torch.distributions import constraints
from torch.distributions.utils import _sum_rightmost

from pyro.distributions.torch_distribution import TorchDistribution


class CensoredDistribution(TorchDistribution):

    def __init__(self, base_distribution, upper_lim=float('inf'), lower_lim=float('-inf'), validate_args=None):
        # Log-prob only computed correctly for univariate base distribution
        assert base_distribution.event_dim == 0 or base_distribution.event_dim == 1 and base_distribution.event_shape[0] == 1
        self.base_dist = base_distribution
        self.upper_lim = upper_lim
        self.lower_lim = lower_lim

        super(CensoredDistribution, self).__init__(self.base_dist.batch_shape, self.base_dist.event_shape,
                                                   validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        raise NotImplemented

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            x[x > self.upper_lim] = self.upper_lim
            x[x < self.lower_lim] = self.lower_lim
            return x

    def rsample(self, sample_shape=torch.Size()):
        x = self.base_dist.sample(sample_shape)
        x[x > self.upper_lim] = self.upper_lim
        x[x < self.lower_lim] = self.lower_lim


    def log_prob(self, value):
        """
        Scores the sample by giving a probability density relative to a new base measure.
        The new base measure places an atom at `self.upper_lim` and `self.lower_lim`, and
        has Lebesgue measure on the intervening interval.

        Thus, `log_prob(self.lower_lim)` and `log_prob(self.upper_lim)` represent probabilities
        as for discrete distributions. `log_prob(x)` in the interior represent regular
        pdfs with respect to Lebesgue measure on R.

        **Note**: `log_prob` scores from distributions with different censoring are not 
        comparable.
        """
        log_prob = self.base_dist.log_prob(value)
        upper_cdf = 1. - self.base_dist.cdf(self.upper_lim)
        lower_cdf = self.base_dist.cdf(self.lower_lim)

        log_prob[value == self.upper_lim] = torch.log(upper_cdf.expand_as(log_prob)[value == self.upper_lim])
        log_prob[value > self.upper_lim] = float('-inf')
        log_prob[value == self.lower_lim] = torch.log(lower_cdf.expand_as(log_prob)[value == self.lower_lim])
        log_prob[value < self.lower_lim] = float('-inf')

        return log_prob

    def cdf(self, value):
        if self._validate_args:
            self.base_dist._validate_sample(value)
        cdf = self.base_dist.cdf(value)
        cdf[value >= self.upper_lim] = 1.
        cdf[value < self.lower_lim] = 0.

    def icdf(self, value):
        # Is this even possible?
        raise NotImplemented

