from __future__ import absolute_import, division, print_function

import operator

from six.moves import reduce
import torch
from torch.distributions import constraints

from pyro.distributions.torch import Categorical
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(TorchDistribution)
class Empirical(TorchDistribution):
    r"""
    Empirical distribution associated with the sampled data. Note that the shape
    requirement for `log_weights` is that its leftmost shape must match that of
    `samples`. Samples are aggregated along the ``aggregation_dim``, which is the
    rightmost dim of `log_weights`.

    e.g. If ``samples.shape = torch.Size([2, 3, 10])`` and
    ``log_weights.shape = torch.Size([2, 3])``, the second dimension corresponds
    to the `aggregation_dim`. The distribution's `batch_shape` is ``torch.Size([2])``
    and its `event_shape` is ``torch.Size([10])``. While sampling, we generate
    a batch of random indices amongst ``[0, 1, 2]``, which are used to index
    into the aggregation dim to return samples of shape ``torch.Size([2, 10])``.

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
        super(TorchDistribution, self).__init__(batch_shape=weight_shape[:-1],
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
        num_samples = reduce(operator.mul, sample_shape, 1)
        dim_order = list(range(self._samples.dim()))
        dim_order.insert(self._aggregation_dim, dim_order.pop(0))
        # If the stored tensors have shape [s_0, s_1, .., s_{agg_dim}, .., s_{n-1}, s_{n}],
        # `sample_idx` must have shape [s_0, s_1, .., num_samples, .., s_{n-1}, s_{n}],
        # wherein we gather `num_samples` values from the aggregation_dim using the indices
        # specified by `sample_idx`.
        sample_idx = self._categorical.sample([num_samples])
        for _ in range(len(self._samples.shape) - sample_idx.dim()):
            sample_idx = sample_idx.unsqueeze(-1)
        sample_idx = sample_idx.permute(dim_order)
        sample_idx = sample_idx.expand(self.batch_shape + torch.Size([-1]) + self.event_shape)
        samples = self._samples.gather(self._aggregation_dim, sample_idx)
        # At this point, samples have `num_samples` values at the aggregation dim.
        # Permute the ordering (and reshape) so that `sample_shape` is leftmost.
        # dim_order.insert(self._aggregation_dim, dim_order.pop(0))
        return samples.permute(dim_order).reshape(sample_shape + self.batch_shape + self.event_shape)

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
