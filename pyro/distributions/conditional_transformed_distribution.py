import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.transforms import Transform
from torch.distributions.utils import _sum_rightmost
from pyro.distributions import ConditionalTransform


class ConditionalTransformedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies a sequence of Transforms
    to a base distribution.  Let f be the composition of transforms applied::

        X ~ BaseDistribution
        Y = f(X) ~ ConditionalTransformedDistribution(BaseDistribution, f)
        log p(Y | Z=z) = log p(X | Z=z) + log |det (d(X | Z=z)/dY)|

    Note that the ``.event_shape`` of a :class:`ConditionalTransformedDistribution` is the
    maximum shape of its base distribution and its transforms, since transforms
    can introduce correlations among events.

    """
    arg_constraints = {}

    def __init__(self, base_distribution, transforms, validate_args=None):
        self.base_dist = base_distribution
        if isinstance(transforms, Transform):
            self.transforms = [transforms, ]
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) or isinstance(t, ConditionalTransform) for t in transforms):
                raise ValueError("transforms must be a Transform or a list of Transforms")
            self.transforms = transforms
        else:
            raise ValueError("transforms must be a Transform or list, but was {}".format(transforms))
        shape = self.base_dist.batch_shape + self.base_dist.event_shape
        event_dim = max([len(self.base_dist.event_shape)] + [t.event_dim for t in self.transforms])
        batch_shape = shape[:len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim:]
        super(ConditionalTransformedDistribution, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ConditionalTransformedDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        base_dist_batch_shape = batch_shape + self.base_dist.batch_shape[len(self.batch_shape):]
        new.base_dist = self.base_dist.expand(base_dist_batch_shape)
        new.transforms = self.transforms
        super(ConditionalTransformedDistribution, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property
    def support(self):
        return self.transforms[-1].codomain if self.transforms else self.base_dist.support

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    def sample(self, obs, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                if isinstance(transform, ConditionalTransform):
                    x = transform(x, obs)
                else:
                    x = transform(x)
            return x

    def rsample(self, obs, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                x = transform(x, obs)
            else:
                x = transform(x)
        return x

    def log_prob(self, value, obs):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        event_dim = len(self.event_shape)
        log_prob = 0.0
        y = value
        for transform in reversed(self.transforms):
            if isinstance(transform, ConditionalTransform):
                x = transform.inv(y)
                log_prob = log_prob - _sum_rightmost(transform.log_abs_det_jacobian(x, y, obs),
                                                     event_dim - transform.event_dim)
            else:
                x = transform.inv(y, obs)
                log_prob = log_prob - _sum_rightmost(transform.log_abs_det_jacobian(x, y),
                                                     event_dim - transform.event_dim)

            y = x

        # NOTE: Conditional base distributions not yet possible. Implement as a transform on base distribution!
        log_prob = log_prob + _sum_rightmost(self.base_dist.log_prob(y),
                                             event_dim - len(self.base_dist.event_shape))
        return log_prob

    def _monotonize_cdf(self, value):
        """
        This conditionally flips ``value -> 1-value`` to ensure :meth:`cdf` is
        monotone increasing.
        """
        sign = 1
        for transform in self.transforms:
            sign = sign * transform.sign
        if isinstance(sign, int) and sign == 1:
            return value
        return sign * (value - 0.5) + 0.5

    def cdf(self, value, obs):
        """
        Computes the cumulative distribution function by inverting the
        transform(s) and computing the score of the base distribution.
        """
        for transform in self.transforms[::-1]:
            if isinstance(transform, ConditionalTransform):
                value = transform.inv(value, obs)
            else:
                value = transform.inv(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.cdf(value)
        value = self._monotonize_cdf(value)
        return value

    def icdf(self, value, obs):
        """
        Computes the inverse cumulative distribution function using
        transform(s) and computing the score of the base distribution.
        """
        value = self._monotonize_cdf(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.icdf(value)
        for transform in self.transforms:
            if isinstance(transform, ConditionalTransform):
                value = transform(value, obs)
            else:
                value = transform(value)
        return value
