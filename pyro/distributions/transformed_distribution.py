import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.transforms import Transform
from torch.distributions.utils import _sum_rightmost


class TransformedDistribution(Distribution):
    r"""
    Extension of the Distribution class, which applies a sequence of Transforms
    to a base distribution.  Let f be the composition of transforms applied::

        X ~ BaseDistribution
        Y = f(X) ~ TransformedDistribution(BaseDistribution, f)
        log p(Y) = log p(X) + log |det (dX/dY)|

    Note that the ``.event_shape`` of a :class:`TransformedDistribution` is the
    maximum shape of its base distribution and its transforms - possibly allowing
    for reshaping - since transforms can introduce correlations among events.

    Example usage:

    >>> # Building a Logistic Distribution
    >>> # X ~ Uniform(0, 1)
    >>> # f = a + b * logit(X)
    >>> # Y ~ f(X) ~ Logistic(a, b)
    >>> base_distribution = Uniform(0, 1)
    >>> transforms = [SigmoidTransform().inv, AffineTransform(loc=a, scale=b)]
    >>> logistic = TransformedDistribution(base_distribution, transforms)

    More examples of constructing distributions from transforms can be found in
    `torch.distributions` and `pyro.distributions`.

    """
    arg_constraints = {}

    def __init__(self, base_distribution, transforms, validate_args=None):
        self.base_dist = base_distribution
        if isinstance(transforms, Transform):
            self.transforms = [transforms, ]
        elif isinstance(transforms, list):
            if not all(isinstance(t, Transform) for t in transforms):
                raise ValueError("transforms must be a Transform or a list of Transforms")
            self.transforms = transforms
        else:
            raise ValueError("transforms must be a Transform or list, but was {}".format(transforms))

        # Calculate shape and event_dim of base distribution (before transforms potentially introduce correlations)
        self.base_shape = self.base_dist.batch_shape + self.base_dist.event_shape
        shape = self.base_shape
        self.event_dim = self.base_dist.event_dim

        # Calculate these values for all transforms
        x = self.base_dist.sample()
        for t in self.transforms:
            # TODO: Validate that domains and ranges match up

            # Try to run forward operation of transform
            try:
                x = t(x)

            # Transform may not have a forward operator defined, in which case assume that it doesn't change shape
            except NotImplementedError:
                x = constraints.transform_to(t.codomain)(torch.zeros(*shape))
                x_inv = t._inverse(x)
                if x_inv.size() != x.size():
                    raise ValueError("Transform with undefined forward operation changes shape of inverse!")

            # Work out the event dimension so far
            # self.event_dim is always relative to the base distribution!
            # len(base_shape) - len(x.size()) = the amount of contraction/expansion due to reshaping
            correlated_dims = t.event_dim + (len(self.base_shape) - len(x.size()))
            self.event_dim = max(self.event_dim, correlated_dims)

            # This could happen if Transform object incorrectly defines t.event_dim
            if correlated_dims < 0:
                raise ValueError("Output of transform has < 0 correlated dimensions!")

            # Check that event dim count is valid, i.e., that it isn't greater than original count of batch +
            # event shapes. You can convert a batch dimension into an event dimension, but you can't exceed the
            # base number.
            if self.event_dim > len(self.base_shape):
                raise ValueError("Event dimension exceeds size of the base distribution shape!")

            # Check that dimensions haven't been introduced or removed
            if torch.tensor(self.base_shape).prod() != torch.tensor(x.size()).prod():
                raise ValueError(
                    "There is a mismatch between the base input shape {} and the output shape {} \
                        of the Transform!".format(self.base_shape, x.size()))

            shape = x.size()

        batch_dims = len(self.base_shape) - self.event_dim
        batch_shape = shape[:batch_dims]
        event_shape = shape[batch_dims:]
        super(TransformedDistribution, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(TransformedDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        batch_dims = len(self.base_shape) - self.event_dim
        base_dist_batch_shape = batch_shape + self.base_dist.batch_shape[batch_dims:]
        new.base_dist = self.base_dist.expand(base_dist_batch_shape)
        new.transforms = self.transforms
        super(TransformedDistribution, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property
    def support(self):
        return self.transforms[-1].codomain if self.transforms else self.base_dist.support

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    def sample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            for transform in self.transforms:
                x = transform(x)
            return x

    def rsample(self, sample_shape=torch.Size()):
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x

    def log_prob(self, value):
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        # Number of sample dimensions, which is only known at run-time (i.e. not an intrisic property of a
        # distribution). This uses max since value may broadcast on its leftmost dimension
        sample_dims = max(0, len(value.shape) - len(self.batch_shape) - len(self.event_shape))

        # Apply transforms in reverse to get sample from base distribution
        log_prob = 0.0
        y = value
        for idx, transform in enumerate(reversed(self.transforms)):
            x = transform.inv(y)

            # We would normally sum out the rightmost dimensions due to the discrepancy between the output
            # event_shape and the event_shape of the input to the transform. But we have to correct by a
            # term due to potential reshapes!
            contracted_dims = len(self.base_shape) - len(y.size()) + sample_dims
            sum_dims = self.event_dim - (transform.event_dim + contracted_dims)
            log_prob = log_prob - _sum_rightmost(transform.log_abs_det_jacobian(x, y), sum_dims)
            y = x

        log_prob = log_prob + _sum_rightmost(self.base_dist.log_prob(y),
                                             self.event_dim - len(self.base_dist.event_shape))
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

    def cdf(self, value):
        """
        Computes the cumulative distribution function by inverting the
        transform(s) and computing the score of the base distribution.
        """
        for transform in self.transforms[::-1]:
            value = transform.inv(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.cdf(value)
        value = self._monotonize_cdf(value)
        return value

    def icdf(self, value):
        """
        Computes the inverse cumulative distribution function using
        transform(s) and computing the score of the base distribution.
        """
        value = self._monotonize_cdf(value)
        if self._validate_args:
            self.base_dist._validate_sample(value)
        value = self.base_dist.icdf(value)
        for transform in self.transforms:
            value = transform(value)
        return value
