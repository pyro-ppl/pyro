# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints

from pyro.distributions.torch import Categorical
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Empirical(TorchDistribution):
    r"""
    Empirical distribution associated with the sampled data. Note that the shape
    requirement for `log_weights` is that its shape must match the leftmost shape
    of `samples`. Samples are aggregated along the ``aggregation_dim``, which is
    the rightmost dim of `log_weights`.

    Example:

    >>> emp_dist = Empirical(torch.randn(2, 3, 10), torch.ones(2, 3))
    >>> emp_dist.batch_shape
    torch.Size([2])
    >>> emp_dist.event_shape
    torch.Size([10])

    >>> single_sample = emp_dist.sample()
    >>> single_sample.shape
    torch.Size([2, 10])
    >>> batch_sample = emp_dist.sample((100,))
    >>> batch_sample.shape
    torch.Size([100, 2, 10])

    >>> emp_dist.log_prob(single_sample).shape
    torch.Size([2])
    >>> # Vectorized samples cannot be scored by log_prob.
    >>> with pyro.validation_enabled():
    ...     emp_dist.log_prob(batch_sample).shape
    Traceback (most recent call last):
    ...
    ValueError: ``value.shape`` must be torch.Size([2, 10])

    :param torch.Tensor samples: samples from the empirical distribution.
    :param torch.Tensor log_weights: log weights (optional) corresponding
        to the samples.
    """

    arg_constraints = {}
    support = constraints.real
    has_enumerate_support = True

    def __init__(self, samples, log_weights, validate_args=None):
        self._samples = samples
        self._log_weights = log_weights
        sample_shape, weight_shape = samples.size(), log_weights.size()
        if weight_shape > sample_shape or weight_shape != sample_shape[:len(weight_shape)]:
            raise ValueError("The shape of ``log_weights`` ({}) must match "
                             "the leftmost shape of ``samples`` ({})".format(weight_shape, sample_shape))
        self._aggregation_dim = log_weights.dim() - 1
        event_shape = sample_shape[len(weight_shape):]
        self._categorical = Categorical(logits=self._log_weights)
        super().__init__(batch_shape=weight_shape[:-1],
                         event_shape=event_shape,
                         validate_args=validate_args)

    @property
    def sample_size(self):
        """
        Number of samples that constitute the empirical distribution.

        :return int: number of samples collected.
        """
        return self._log_weights.numel()

    def sample(self, sample_shape=torch.Size()):
        sample_idx = self._categorical.sample(sample_shape)  # sample_shape x batch_shape
        # reorder samples to bring aggregation_dim to the front:
        # batch_shape x num_samples x event_shape -> num_samples x batch_shape x event_shape
        samples = self._samples.unsqueeze(0).transpose(0, self._aggregation_dim + 1).squeeze(self._aggregation_dim + 1)
        # make sample_idx.shape compatible with samples.shape: sample_shape_numel x batch_shape x event_shape
        sample_idx = sample_idx.reshape((-1,) + self.batch_shape + (1,) * len(self.event_shape))
        sample_idx = sample_idx.expand((-1,) + samples.shape[1:])
        return samples.gather(0, sample_idx).reshape(sample_shape + samples.shape[1:])

    def log_prob(self, value):
        """
        Returns the log of the probability mass function evaluated at ``value``.
        Note that this currently only supports scoring values with empty
        ``sample_shape``.

        :param torch.Tensor value: scalar or tensor value to be scored.
        """
        if self._validate_args:
            if value.shape != self.batch_shape + self.event_shape:
                raise ValueError("``value.shape`` must be {}".format(self.batch_shape + self.event_shape))
        if self.batch_shape:
            value = value.unsqueeze(self._aggregation_dim)
        selection_mask = self._samples.eq(value)
        # Get a mask for all entries in the ``weights`` tensor
        # that correspond to ``value``.
        for _ in range(len(self.event_shape)):
            selection_mask = selection_mask.min(dim=-1)[0]
        selection_mask = selection_mask.type(self._categorical.probs.type())
        return (self._categorical.probs * selection_mask).sum(dim=-1).log()

    def _weighted_mean(self, value, keepdim=False):
        weights = self._log_weights.reshape(self._log_weights.size() +
                                            torch.Size([1] * (value.dim() - self._log_weights.dim())))
        dim = self._aggregation_dim
        max_weight = weights.max(dim=dim, keepdim=True)[0]
        relative_probs = (weights - max_weight).exp()
        return (value * relative_probs).sum(dim=dim, keepdim=keepdim) / relative_probs.sum(dim=dim, keepdim=keepdim)

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def mean(self):
        if self._samples.dtype in (torch.int32, torch.int64):
            raise ValueError("Mean for discrete empirical distribution undefined. " +
                             "Consider converting samples to ``torch.float32`` " +
                             "or ``torch.float64``. If these are samples from a " +
                             "`Categorical` distribution, consider converting to a " +
                             "`OneHotCategorical` distribution.")
        return self._weighted_mean(self._samples)

    @property
    def variance(self):
        if self._samples.dtype in (torch.int32, torch.int64):
            raise ValueError("Variance for discrete empirical distribution undefined. " +
                             "Consider converting samples to ``torch.float32`` " +
                             "or ``torch.float64``. If these are samples from a " +
                             "`Categorical` distribution, consider converting to a " +
                             "`OneHotCategorical` distribution.")
        mean = self.mean.unsqueeze(self._aggregation_dim)
        deviation_squared = torch.pow(self._samples - mean, 2)
        return self._weighted_mean(deviation_squared)

    @property
    def log_weights(self):
        return self._log_weights

    def enumerate_support(self, expand=True):
        # Empirical does not support batching, so expanding is a no-op.
        return self._samples
