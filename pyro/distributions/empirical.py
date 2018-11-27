from __future__ import absolute_import, division, print_function

import math
import numbers
from collections import defaultdict

import torch
from contextlib2 import contextmanager
from torch.distributions import constraints

from pyro.distributions.torch import Categorical
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Empirical(TorchDistribution):
    r"""
    Empirical distribution associated with the sampled data.

    :param torch.Tensor samples: samples from the empirical distribution.
    :param torch.Tensor log_weights: log weights (optional) corresponding
        to the samples. The leftmost shape of ``log_weights`` must match
        that of samples
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
        self._event_shape = sample_shape[len(weight_shape):]
        self._categorical = Categorical(logits=self._log_weights)
        super(TorchDistribution, self).__init__(batch_shape=weight_shape[:-1],
                                                validate_args=validate_args)

    @property
    def sample_size(self):
        """
        Number of samples that constitute the empirical distribution.

        :return int: number of samples collected.
        """
        return self._log_weights.numel()

    def sample(self, sample_shape=torch.Size()):
        sample_idx = self._categorical.sample(sample_shape)
        return self._samples[sample_idx]

    def log_prob(self, value):
        """
        Returns the log of the probability mass function evaluated at ``value``.
        Note that this currently only supports scoring values with empty
        ``sample_shape``, i.e. an arbitrary batched sample is not allowed.

        :param torch.Tensor value: scalar or tensor value to be scored.
        """
        if self._validate_args:
            if value.shape != self.batch_shape + self.event_shape:
                raise ValueError("``value.shape`` must be {}".format(self.batch_shape + self.event_shape))
        selection_mask = self._samples.eq(value).reshape(self.batch_shape, -1)
        if self.event_shape:
            selection_mask = selection_mask.min(dim=-1)[0]
        return self._categorical.probs.masked_select(selection_mask).sum(dim=-1).log()

    def _weighted_mean(self, value, keepdim=False):
        weights = self._log_weights.reshape(self.weights.size() +
                                            torch.Size([1] * (value.dim() - self.weights.dim())))
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
    def weights(self):
        return self._log_weights

    def enumerate_support(self, expand=True, flatten=True):
        # Empirical does not support batching, so expanding is a no-op.
        return self._samples
