from __future__ import absolute_import, division, print_function

import numbers

import torch
from torch.distributions import biject_to, constraints, transform_to
from torch.distributions.kl import kl_divergence, register_kl

import pyro.distributions.torch
from pyro.distributions.distribution import Distribution
from pyro.distributions.score_parts import ScoreParts
from pyro.distributions.util import broadcast_shape, scale_and_mask, sum_rightmost


class TorchDistributionMixin(Distribution):
    """
    Mixin to provide Pyro compatibility for PyTorch distributions.

    You should instead use `TorchDistribution` for new distribution classes.

    This is mainly useful for wrapping existing PyTorch distributions for
    use in Pyro.  Derived classes must first inherit from
    :class:`torch.distributions.distribution.Distribution` and then inherit
    from :class:`TorchDistributionMixin`.
    """
    def __call__(self, sample_shape=torch.Size()):
        """
        Samples a random value.

        This is reparameterized whenever possible, calling
        :meth:`~torch.distributions.distribution.Distribution.rsample` for
        reparameterized distributions and
        :meth:`~torch.distributions.distribution.Distribution.sample` for
        non-reparameterized distributions.

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: A random value or batch of random values (if parameters are
            batched). The shape of the result should be `self.shape()`.
        :rtype: torch.Tensor
        """
        return self.rsample(sample_shape) if self.has_rsample else self.sample(sample_shape)

    @property
    def event_dim(self):
        """
        :return: Number of dimensions of individual events.
        :rtype: int
        """
        return len(self.event_shape)

    def shape(self, sample_shape=torch.Size()):
        """
        The tensor shape of samples from this distribution.

        Samples are of shape::

          d.shape(sample_shape) == sample_shape + d.batch_shape + d.event_shape

        :param sample_shape: the size of the iid batch to be drawn from the
            distribution.
        :type sample_shape: torch.Size
        :return: Tensor shape of samples.
        :rtype: torch.Size
        """
        return sample_shape + self.batch_shape + self.event_shape

    def expand(self, batch_shape):
        """
        Expands a distribution to a desired
        :attr:`~torch.distributions.distribution.Distribution.batch_shape`.

        Note that this is more general than :meth:`expand_by` because
        ``d.expand_by(sample_shape)`` can be reduced to
        ``d.expand(sample_shape + d.batch_shape)``.

        :param torch.Size batch_shape: The target ``batch_shape``. This must
            compatible with ``self.batch_shape`` similar to the requirements
            of :func:`torch.Tensor.expand`: the target ``batch_shape`` must
            be at least as long as ``self.batch_shape``, and for each
            non-singleton dim of ``self.batch_shape``, ``batch_shape`` must
            either agree or be set to ``-1``.
        :return: An expanded version of this distribution.
        :rtype: :class:`ReshapedDistribution`
        """
        batch_shape = torch.Size(batch_shape)
        cut = len(batch_shape) - len(self.batch_shape)
        left, right = batch_shape[:cut], batch_shape[cut:]
        if right == self.batch_shape:
            return self.expand_by(left)
        else:
            raise NotImplementedError("`TorchDistributionMixin.expand()` cannot expand "
                                      "distribution's existing batch shape. Consider "
                                      "overriding the default implementation for the "
                                      "distribution class.")

    def expand_by(self, sample_shape):
        """
        Expands a distribution by adding ``sample_shape`` to the left side of
        its :attr:`~torch.distributions.distribution.Distribution.batch_shape`.

        To expand internal dims of ``self.batch_shape`` from 1 to something
        larger, use :meth:`expand` instead.

        :param torch.Size sample_shape: The size of the iid batch to be drawn
            from the distribution.
        :return: An expanded version of this distribution.
        :rtype: :class:`ReshapedDistribution`
        """
        if not sample_shape:
            return self
        return ReshapedDistribution(self, sample_shape=sample_shape)

    def reshape(self, sample_shape=None, extra_event_dims=None):
        raise Exception('''
            .reshape(sample_shape=s, extra_event_dims=n) was renamed and split into
            .expand_by(sample_shape=s).independent(reinterpreted_batch_ndims=n).''')

    def independent(self, reinterpreted_batch_ndims=None):
        """
        Reinterprets the ``n`` rightmost dimensions of this distributions
        :attr:`~torch.distributions.distribution.Distribution.batch_shape`
        as event dims, adding them to the left side of
        :attr:`~torch.distributions.distribution.Distribution.event_shape`.

        Example:

            .. doctest::
               :hide:

               >>> d0 = dist.Normal(torch.zeros(2, 3, 4, 5), torch.ones(2, 3, 4, 5))
               >>> [d0.batch_shape, d0.event_shape]
               [torch.Size([2, 3, 4, 5]), torch.Size([])]
               >>> d1 = d0.independent(2)

            >>> [d1.batch_shape, d1.event_shape]
            [torch.Size([2, 3]), torch.Size([4, 5])]
            >>> d2 = d1.independent(1)
            >>> [d2.batch_shape, d2.event_shape]
            [torch.Size([2]), torch.Size([3, 4, 5])]
            >>> d3 = d1.independent(2)
            >>> [d3.batch_shape, d3.event_shape]
            [torch.Size([]), torch.Size([2, 3, 4, 5])]

        :param int reinterpreted_batch_ndims: The number of batch dimensions
            to reinterpret as event dimensions.
        :return: A reshaped version of this distribution.
        :rtype: :class:`pyro.distributions.torch.Independent`
        """
        if reinterpreted_batch_ndims is None:
            reinterpreted_batch_ndims = len(self.batch_shape)
        return pyro.distributions.torch.Independent(self, reinterpreted_batch_ndims)

    def mask(self, mask):
        """
        Masks a distribution by a zero-one tensor that is broadcastable to the
        distributions :attr:`~torch.distributions.distribution.Distribution.batch_shape`.

        :param torch.Tensor mask: A zero-one valued float tensor.
        :return: A masked copy of this distribution.
        :rtype: :class:`MaskedDistribution`
        """
        return MaskedDistribution(self, mask)


class TorchDistribution(torch.distributions.Distribution, TorchDistributionMixin):
    """
    Base class for PyTorch-compatible distributions with Pyro support.

    This should be the base class for almost all new Pyro distributions.

    .. note::

        Parameters and data should be of type :class:`~torch.Tensor`
        and all methods return type :class:`~torch.Tensor` unless
        otherwise noted.

    **Tensor Shapes**:

    TorchDistributions provide a method ``.shape()`` for the tensor shape of samples::

      x = d.sample(sample_shape)
      assert x.shape == d.shape(sample_shape)

    Pyro follows the same distribution shape semantics as PyTorch. It distinguishes
    between three different roles for tensor shapes of samples:

    - *sample shape* corresponds to the shape of the iid samples drawn from the distribution.
      This is taken as an argument by the distribution's `sample` method.
    - *batch shape* corresponds to non-identical (independent) parameterizations of
      the distribution, inferred from the distribution's parameter shapes. This is
      fixed for a distribution instance.
    - *event shape* corresponds to the event dimensions of the distribution, which
      is fixed for a distribution class. These are collapsed when we try to score
      a sample from the distribution via `d.log_prob(x)`.

    These shapes are related by the equation::

      assert d.shape(sample_shape) == sample_shape + d.batch_shape + d.event_shape

    Distributions provide a vectorized
    :meth:`~torch.distributions.distribution.Distribution.log_prob` method that
    evaluates the log probability density of each event in a batch
    independently, returning a tensor of shape
    ``sample_shape + d.batch_shape``::

      x = d.sample(sample_shape)
      assert x.shape == d.shape(sample_shape)
      log_p = d.log_prob(x)
      assert log_p.shape == sample_shape + d.batch_shape

    **Implementing New Distributions**:

    Derived classes must implement the methods
    :meth:`~torch.distributions.distribution.Distribution.sample`
    (or :meth:`~torch.distributions.distribution.Distribution.rsample` if
    ``.has_rsample == True``) and
    :meth:`~torch.distributions.distribution.Distribution.log_prob`, and must
    implement the properties
    :attr:`~torch.distributions.distribution.Distribution.batch_shape`,
    and :attr:`~torch.distributions.distribution.Distribution.event_shape`.
    Discrete classes may also implement the
    :meth:`~torch.distributions.distribution.Distribution.enumerate_support`
    method to improve gradient estimates and set
    ``.has_enumerate_support = True``.
    """
    pass


# TODO move this upstream to torch.distributions
class IndependentConstraint(constraints.Constraint):
    """
    Wraps a constraint by aggregating over ``reinterpreted_batch_ndims``-many
    dims in :meth:`check`, so that an event is valid only if all its
    independent entries are valid.

    :param torch.distributions.constraints.Constraint base_constraint: A base
        constraint whose entries are incidentally indepenent.
    :param int reinterpreted_batch_ndims: The number of extra event dimensions that will
        be considered dependent.
    """
    def __init__(self, base_constraint, reinterpreted_batch_ndims):
        self.base_constraint = base_constraint
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims

    def check(self, value):
        result = self.base_constraint.check(value)
        result = result.reshape(result.shape[:result.dim() - self.reinterpreted_batch_ndims] + (-1,))
        result = result.min(-1)[0]
        return result


biject_to.register(IndependentConstraint, lambda c: biject_to(c.base_constraint))
transform_to.register(IndependentConstraint, lambda c: transform_to(c.base_constraint))


class ReshapedDistribution(TorchDistribution):
    """
    Reshapes a distribution by adding ``sample_shape`` to its total shape
    and adding ``reinterpreted_batch_ndims`` to its
    :attr:`~torch.distributions.distribution.Distribution.event_shape`.

    :param torch.Size sample_shape: The size of the iid batch to be drawn from
        the distribution.
    :param int reinterpreted_batch_ndims: The number of extra event dimensions that will
        be considered dependent.
    """
    arg_constraints = {}

    def __init__(self, base_dist, sample_shape=torch.Size(), reinterpreted_batch_ndims=0):
        sample_shape = torch.Size(sample_shape)
        if reinterpreted_batch_ndims > len(sample_shape + base_dist.batch_shape):
            raise ValueError('Expected reinterpreted_batch_ndims <= len(sample_shape + base_dist.batch_shape), '
                             'actual {} vs {}'.format(reinterpreted_batch_ndims,
                                                      len(sample_shape + base_dist.batch_shape)))
        self.base_dist = base_dist
        self.sample_shape = sample_shape
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        shape = sample_shape + base_dist.batch_shape + base_dist.event_shape
        batch_dim = len(shape) - reinterpreted_batch_ndims - len(base_dist.event_shape)
        batch_shape, event_shape = shape[:batch_dim], shape[batch_dim:]
        super(ReshapedDistribution, self).__init__(batch_shape, event_shape)

    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        # Raise error if existing batch shape is being shrunk.
        # e.g. (2, 4) -> (2, 1)
        proposed_shape = broadcast_shape(self.batch_shape, batch_shape)
        if tuple(reversed(proposed_shape)) > tuple(reversed(batch_shape)):
            raise ValueError("Existing batch shape {} cannot be expanded "
                             "to the new batch shape {}."
                             .format(self.batch_shape, batch_shape))
        # Adjust existing sample shape if possible.
        base_dist = self.base_dist
        base_batch_shape = batch_shape + self.event_shape[:self.reinterpreted_batch_ndims]
        cut = len(base_batch_shape) - len(base_dist.batch_shape)
        left, right = base_batch_shape[:cut], base_batch_shape[cut:]
        if right == base_dist.batch_shape:
            sample_shape = left
        # Modify the base distribution's batch shape,
        # if existing sample shape cannot be adjusted.
        else:
            base_dist = self.base_dist.expand(base_batch_shape)
            assert not isinstance(base_dist, ReshapedDistribution)
            sample_shape = torch.Size(())
        return ReshapedDistribution(base_dist, sample_shape, self.reinterpreted_batch_ndims)

    def expand_by(self, sample_shape):
        base_dist = self.base_dist
        sample_shape = torch.Size(sample_shape) + self.sample_shape
        reinterpreted_batch_ndims = self.reinterpreted_batch_ndims
        return ReshapedDistribution(base_dist, sample_shape, reinterpreted_batch_ndims)

    def independent(self, reinterpreted_batch_ndims=None):
        if reinterpreted_batch_ndims is None:
            reinterpreted_batch_ndims = len(self.batch_shape)
        base_dist = self.base_dist
        sample_shape = self.sample_shape
        reinterpreted_batch_ndims = self.reinterpreted_batch_ndims + reinterpreted_batch_ndims
        return ReshapedDistribution(base_dist, sample_shape, reinterpreted_batch_ndims)

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    def support(self):
        return IndependentConstraint(self.base_dist.support, self.reinterpreted_batch_ndims)

    @property
    def _validate_args(self):
        return self.base_dist._validate_args

    @_validate_args.setter
    def _validate_args(self, value):
        self.base_dist._validate_args = value

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape + self.sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape + self.sample_shape)

    def log_prob(self, value):
        shape = broadcast_shape(self.batch_shape, value.shape[:value.dim() - self.event_dim])
        return sum_rightmost(self.base_dist.log_prob(value), self.reinterpreted_batch_ndims).expand(shape)

    def score_parts(self, value):
        shape = broadcast_shape(self.batch_shape, value.shape[:value.dim() - self.event_dim])
        log_prob, score_function, entropy_term = self.base_dist.score_parts(value)
        log_prob = sum_rightmost(log_prob, self.reinterpreted_batch_ndims).expand(shape)
        if not isinstance(score_function, numbers.Number):
            score_function = sum_rightmost(score_function, self.reinterpreted_batch_ndims).expand(shape)
        if not isinstance(entropy_term, numbers.Number):
            entropy_term = sum_rightmost(entropy_term, self.reinterpreted_batch_ndims).expand(shape)
        return ScoreParts(log_prob, score_function, entropy_term)

    def enumerate_support(self, expand=True):
        if self.reinterpreted_batch_ndims:
            raise NotImplementedError("Pyro does not enumerate over cartesian products")

        samples = self.base_dist.enumerate_support(expand=False)
        samples = samples.reshape(samples.shape[:1] + (1,) * len(self.batch_shape) + self.event_shape)
        if expand:
            samples = samples.expand(samples.shape[:1] + self.batch_shape + self.event_shape)
        return samples

    @property
    def mean(self):
        return self.base_dist.mean.expand(self.batch_shape + self.event_shape)

    @property
    def variance(self):
        return self.base_dist.variance.expand(self.batch_shape + self.event_shape)

    def entropy(self):
        return sum_rightmost(self.base_dist.entropy(), self.reinterpreted_batch_ndims)


@register_kl(ReshapedDistribution, ReshapedDistribution)
def _kl_reshaped_reshaped(p, q):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError
    kl = kl_divergence(p.base_dist, q.base_dist)
    if p.reinterpreted_batch_ndims:
        kl = sum_rightmost(kl, p.reinterpreted_batch_ndims)
    shape = broadcast_shape(p.batch_shape, q.batch_shape)
    return kl.expand(shape)


class MaskedDistribution(TorchDistribution):
    """
    Masks a distribution by a zero-one tensor that is broadcastable to the
    distribution's :attr:`~torch.distributions.distribution.Distribution.batch_shape`.

    :param torch.Tensor mask: A zero-one valued tensor.
    """
    arg_constraints = {}

    def __init__(self, base_dist, mask):
        if broadcast_shape(mask.shape, base_dist.batch_shape) != base_dist.batch_shape:
            raise ValueError("Expected mask.shape to be broadcastable to base_dist.batch_shape, "
                             "actual {} vs {}".format(mask.shape, base_dist.batch_shape))
        self.base_dist = base_dist
        self._mask = mask.byte()
        super(MaskedDistribution, self).__init__(base_dist.batch_shape, base_dist.event_shape)

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    def sample(self, sample_shape=torch.Size()):
        return self.base_dist.sample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self.base_dist.rsample(sample_shape)

    def log_prob(self, value):
        return scale_and_mask(self.base_dist.log_prob(value), mask=self._mask)

    def score_parts(self, value):
        return self.base_dist.score_parts(value).scale_and_mask(mask=self._mask)

    def enumerate_support(self, expand=True):
        return self.base_dist.enumerate_support(expand=expand)

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance


@register_kl(MaskedDistribution, MaskedDistribution)
def _kl_masked_masked(p, q):
    mask = p._mask if p._mask is q._mask else p._mask & q._mask
    kl = kl_divergence(p.base_dist, q.base_dist)
    return scale_and_mask(kl, mask=mask)
