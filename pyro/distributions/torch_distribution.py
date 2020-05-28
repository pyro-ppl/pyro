# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections import OrderedDict

import torch
from torch.distributions import constraints
from torch.distributions.kl import kl_divergence, register_kl

import pyro.distributions.torch
from pyro.distributions.distribution import Distribution
from pyro.distributions.score_parts import ScoreParts
from pyro.distributions.util import broadcast_shape, scale_and_mask


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

    def expand(self, batch_shape, _instance=None):
        """
        Returns a new :class:`ExpandedDistribution` instance with batch
        dimensions expanded to `batch_shape`.

        :param tuple batch_shape: batch shape to expand to.
        :param _instance: unused argument for compatibility with
            :meth:`torch.distributions.Distribution.expand`
        :return: an instance of `ExpandedDistribution`.
        :rtype: :class:`ExpandedDistribution`
        """
        return ExpandedDistribution(self, batch_shape)

    def expand_by(self, sample_shape):
        """
        Expands a distribution by adding ``sample_shape`` to the left side of
        its :attr:`~torch.distributions.distribution.Distribution.batch_shape`.

        To expand internal dims of ``self.batch_shape`` from 1 to something
        larger, use :meth:`expand` instead.

        :param torch.Size sample_shape: The size of the iid batch to be drawn
            from the distribution.
        :return: An expanded version of this distribution.
        :rtype: :class:`ExpandedDistribution`
        """
        try:
            expanded_dist = self.expand(torch.Size(sample_shape) + self.batch_shape)
        except NotImplementedError:
            expanded_dist = TorchDistributionMixin.expand(self, torch.Size(sample_shape) + self.batch_shape)
        return expanded_dist

    def reshape(self, sample_shape=None, extra_event_dims=None):
        raise Exception('''
            .reshape(sample_shape=s, extra_event_dims=n) was renamed and split into
            .expand_by(sample_shape=s).to_event(reinterpreted_batch_ndims=n).''')

    def to_event(self, reinterpreted_batch_ndims=None):
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
               >>> d1 = d0.to_event(2)

            >>> [d1.batch_shape, d1.event_shape]
            [torch.Size([2, 3]), torch.Size([4, 5])]
            >>> d2 = d1.to_event(1)
            >>> [d2.batch_shape, d2.event_shape]
            [torch.Size([2]), torch.Size([3, 4, 5])]
            >>> d3 = d1.to_event(2)
            >>> [d3.batch_shape, d3.event_shape]
            [torch.Size([]), torch.Size([2, 3, 4, 5])]

        :param int reinterpreted_batch_ndims: The number of batch dimensions to
            reinterpret as event dimensions. May be negative to remove
            dimensions from an :class:`pyro.distributions.torch.Independent` .
            If None, convert all dimensions to event dimensions.
        :return: A reshaped version of this distribution.
        :rtype: :class:`pyro.distributions.torch.Independent`
        """
        if reinterpreted_batch_ndims is None:
            reinterpreted_batch_ndims = len(self.batch_shape)

        # Deconstruct Independent distributions.
        base_dist = self
        while isinstance(base_dist, torch.distributions.Independent):
            reinterpreted_batch_ndims += base_dist.reinterpreted_batch_ndims
            base_dist = base_dist.base_dist

        if reinterpreted_batch_ndims == 0:
            return base_dist
        if reinterpreted_batch_ndims < 0:
            raise ValueError("Cannot remove event dimensions from {}".format(type(self)))
        return pyro.distributions.torch.Independent(base_dist, reinterpreted_batch_ndims)

    def independent(self, reinterpreted_batch_ndims=None):
        warnings.warn("independent is deprecated; use to_event instead", DeprecationWarning)
        return self.to_event(reinterpreted_batch_ndims=reinterpreted_batch_ndims)

    def mask(self, mask):
        """
        Masks a distribution by a boolean or boolean-valued tensor that is
        broadcastable to the distributions
        :attr:`~torch.distributions.distribution.Distribution.batch_shape` .

        :param mask: A boolean or boolean valued tensor.
        :type mask: bool or torch.Tensor
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
    # Provides a default `.expand` method for Pyro distributions which overrides
    # torch.distributions.Distribution.expand (throws a NotImplementedError).
    expand = TorchDistributionMixin.expand


class MaskedDistribution(TorchDistribution):
    """
    Masks a distribution by a boolean tensor that is broadcastable to the
    distribution's :attr:`~torch.distributions.distribution.Distribution.batch_shape`.

    In the special case ``mask is False``, computation of :meth:`log_prob` ,
    :meth:`score_parts` , and ``kl_divergence()`` is skipped, and constant zero
    values are returned instead.

    :param mask: A boolean or boolean-valued tensor.
    :type mask: torch.Tensor or bool
    """
    arg_constraints = {}

    def __init__(self, base_dist, mask):
        if isinstance(mask, bool):
            self._mask = mask
        else:
            batch_shape = broadcast_shape(mask.shape, base_dist.batch_shape)
            if mask.shape != batch_shape:
                mask = mask.expand(batch_shape)
            if base_dist.batch_shape != batch_shape:
                base_dist = base_dist.expand(batch_shape)
            self._mask = mask.bool()
        self.base_dist = base_dist
        super().__init__(base_dist.batch_shape, base_dist.event_shape)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MaskedDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.base_dist = self.base_dist.expand(batch_shape)
        new._mask = self._mask
        if isinstance(new._mask, torch.Tensor):
            new._mask = new._mask.expand(batch_shape)
        super(MaskedDistribution, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

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
        if self._mask is False:
            shape = broadcast_shape(self.base_dist.batch_shape,
                                    value.shape[:value.dim() - self.event_dim])
            return torch.zeros((), device=value.device).expand(shape)
        if self._mask is True:
            return self.base_dist.log_prob(value)
        return scale_and_mask(self.base_dist.log_prob(value), mask=self._mask)

    def score_parts(self, value):
        if isinstance(self._mask, bool):
            return super().score_parts(value)  # calls self.log_prob(value)
        return self.base_dist.score_parts(value).scale_and_mask(mask=self._mask)

    def enumerate_support(self, expand=True):
        return self.base_dist.enumerate_support(expand=expand)

    @property
    def mean(self):
        return self.base_dist.mean

    @property
    def variance(self):
        return self.base_dist.variance

    def conjugate_update(self, other):
        """
        EXPERIMENTAL.
        """
        updated, log_normalizer = self.base_dist.conjugate_update(other)
        updated = updated.mask(self._mask)
        log_normalizer = torch.where(self._mask, log_normalizer, torch.zeros_like(log_normalizer))
        return updated, log_normalizer


class ExpandedDistribution(TorchDistribution):
    arg_constraints = {}

    def __init__(self, base_dist, batch_shape=torch.Size()):
        self.base_dist = base_dist
        super().__init__(base_dist.batch_shape, base_dist.event_shape)
        # adjust batch shape
        self.expand(batch_shape)

    def expand(self, batch_shape, _instance=None):
        # Do basic validation. e.g. we should not "unexpand" distributions even if that is possible.
        new_shape, _, _ = self._broadcast_shape(self.batch_shape, batch_shape)
        # Record interstitial and expanded dims/sizes w.r.t. the base distribution
        new_shape, expanded_sizes, interstitial_sizes = self._broadcast_shape(self.base_dist.batch_shape,
                                                                              new_shape)
        self._batch_shape = new_shape
        self._expanded_sizes = expanded_sizes
        self._interstitial_sizes = interstitial_sizes
        return self

    @staticmethod
    def _broadcast_shape(existing_shape, new_shape):
        if len(new_shape) < len(existing_shape):
            raise ValueError("Cannot broadcast distribution of shape {} to shape {}"
                             .format(existing_shape, new_shape))
        reversed_shape = list(reversed(existing_shape))
        expanded_sizes, interstitial_sizes = [], []
        for i, size in enumerate(reversed(new_shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
                expanded_sizes.append((-i - 1, size))
            elif reversed_shape[i] == 1:
                if size != 1:
                    reversed_shape[i] = size
                    interstitial_sizes.append((-i - 1, size))
            elif reversed_shape[i] != size:
                raise ValueError("Cannot broadcast distribution of shape {} to shape {}"
                                 .format(existing_shape, new_shape))
        return tuple(reversed(reversed_shape)), OrderedDict(expanded_sizes), OrderedDict(interstitial_sizes)

    @property
    def has_rsample(self):
        return self.base_dist.has_rsample

    @property
    def has_enumerate_support(self):
        return self.base_dist.has_enumerate_support

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    def _sample(self, sample_fn, sample_shape):
        interstitial_dims = tuple(self._interstitial_sizes.keys())
        interstitial_dims = tuple(i - self.event_dim for i in interstitial_dims)
        interstitial_sizes = tuple(self._interstitial_sizes.values())
        expanded_sizes = tuple(self._expanded_sizes.values())
        batch_shape = expanded_sizes + interstitial_sizes
        samples = sample_fn(sample_shape + batch_shape)
        interstitial_idx = len(sample_shape) + len(expanded_sizes)
        interstitial_sample_dims = tuple(range(interstitial_idx, interstitial_idx + len(interstitial_sizes)))
        for dim1, dim2 in zip(interstitial_dims, interstitial_sample_dims):
            samples = samples.transpose(dim1, dim2)
        return samples.reshape(sample_shape + self.batch_shape + self.event_shape)

    def sample(self, sample_shape=torch.Size()):
        return self._sample(self.base_dist.sample, sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self._sample(self.base_dist.rsample, sample_shape)

    def log_prob(self, value):
        shape = broadcast_shape(self.batch_shape, value.shape[:value.dim() - self.event_dim])
        log_prob = self.base_dist.log_prob(value)
        return log_prob.expand(shape)

    def score_parts(self, value):
        shape = broadcast_shape(self.batch_shape, value.shape[:value.dim() - self.event_dim])
        log_prob, score_function, entropy_term = self.base_dist.score_parts(value)
        if self.batch_shape != self.base_dist.batch_shape:
            log_prob = log_prob.expand(shape)
            if isinstance(score_function, torch.Tensor):
                score_function = score_function.expand(shape)
            if isinstance(score_function, torch.Tensor):
                entropy_term = entropy_term.expand(shape)
        return ScoreParts(log_prob, score_function, entropy_term)

    def enumerate_support(self, expand=True):
        samples = self.base_dist.enumerate_support(expand=False)
        enum_shape = samples.shape[:1]
        samples = samples.reshape(enum_shape + (1,) * len(self.batch_shape))
        if expand:
            samples = samples.expand(enum_shape + self.batch_shape)
        return samples

    @property
    def mean(self):
        return self.base_dist.mean.expand(self.batch_shape + self.event_shape)

    @property
    def variance(self):
        return self.base_dist.variance.expand(self.batch_shape + self.event_shape)

    def conjugate_update(self, other):
        """
        EXPERIMENTAL.
        """
        updated, log_normalizer = self.base_dist.conjugate_update(other)
        updated = updated.expand(self.batch_shape)
        log_normalizer = log_normalizer.expand(self.batch_shape)
        return updated, log_normalizer


@register_kl(MaskedDistribution, MaskedDistribution)
def _kl_masked_masked(p, q):
    if p._mask is False or q._mask is False:
        mask = False
    elif p._mask is True:
        mask = q._mask
    elif q._mask is True:
        mask = p._mask
    elif p._mask is q._mask:
        mask = p._mask
    else:
        mask = p._mask & q._mask

    if mask is False:
        return 0.  # Return a float, since we cannot determine device.
    if mask is True:
        return kl_divergence(p.base_dist, q.base_dist)
    kl = kl_divergence(p.base_dist, q.base_dist)
    return scale_and_mask(kl, mask=mask)
