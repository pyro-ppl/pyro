from __future__ import absolute_import, division, print_function

import numbers

import math
import torch

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.torch import Categorical
from pyro.distributions.util import copy_docs_from, is_validation_enabled
from pyro.distributions.util import log_sum_exp


@copy_docs_from(TorchDistribution)
class Empirical(TorchDistribution):
    r"""
    Empirical distribution associated with the sampled data.
    """

    arg_constraints = {}

    def __init__(self):
        self._samples = None
        self._log_weights = None
        self._categorical = None
        self._samples_buffer = []
        self._weights_buffer = []
        super(TorchDistribution, self).__init__()

    def _append_from_buffer(self, tensor, buffer):
        """
        Append values from the buffer to the finalized tensor, along the
        leftmost dimension.

        :param torch.Tensor tensor: tensor containing existing values.
        :param list buffer: list of new values.
        :return: tensor with new values appended at the bottom.
        """
        buffer_tensor = torch.stack(buffer, dim=0)
        return torch.cat([tensor, buffer_tensor], dim=0)

    def _finalize(self):
        """
        Appends values collected in the samples/weights buffers to their
        corresponding tensors.
        """
        if not self._samples_buffer:
            return
        self._samples = self._append_from_buffer(self._samples, self._samples_buffer)
        self._log_weights = self._append_from_buffer(self._log_weights, self._weights_buffer)
        self._categorical = Categorical(logits=self._log_weights)
        # Reset buffers.
        self._samples_buffer, self._weights_buffer = [], []

    @property
    def sample_size(self):
        """
        Number of samples that constitute the empirical distribution.

        :return int: number of samples collected.
        """
        self._finalize()
        if self._samples is None:
            return 0
        return self._samples.size(0)

    def add(self, value, log_weight=None, weight=None):
        """
        Adds a new data point to the sample. The values in successive calls to ``add``
        must have the same tensor shape and size. Optionally, an importance weight
        can be specified via ``log_weight`` or ``weight`` (default value of `1`
        is used if not specified).

        :param torch.Tensor value: tensor to add to the sample.
        :param torch.Tensor weight: log weight (optional) corresponding to the sample.
        :param torch.Tensor log_weight: weight (optional) corresponding to the sample.
        """
        if is_validation_enabled():
            if weight is not None and log_weight is not None:
                raise ValueError("Only one of ```weight`` or ``log_weight`` should be specified.")
            if torch.is_tensor(weight) and weight.dim() > 0:
                raise ValueError("``weight.dim() > 0``, but weight should be a scalar.")

        weight_type = torch.Tensor if value.dtype in (torch.int32, torch.int64) else value.type()
        # Apply default weight of 1.0.
        if log_weight is None and weight is None:
            log_weight = torch.tensor(1.0).type(weight_type).log()
        elif weight is not None and log_weight is None:
            log_weight = math.log(weight)
        if isinstance(log_weight, numbers.Number):
            log_weight = torch.tensor(log_weight).type(weight_type)

        # Seed the container tensors with the correct tensor types
        if self._samples is None:
            self._samples = value.new()
            self._log_weights = log_weight.new()
        # Append to the buffer list
        self._samples_buffer.append(value)
        self._weights_buffer.append(log_weight)

    def sample(self, sample_shape=torch.Size()):
        self._finalize()
        idxs = self._categorical.sample(sample_shape=sample_shape)
        return self._samples[idxs]

    def log_prob(self, value):
        """
        Returns the log of the probability mass function evaluated at ``value``.
        Note that this currently only supports scoring single values and not
        a tensor of values.

        :param torch.Tensor value: scalar or tensor value to be scored.
        """
        if is_validation_enabled():
            if value.size() != self.event_shape:
                raise ValueError("``value.size()`` must be {}".format(self.event_shape))
        self._finalize()
        selection_mask = self._samples.eq(value).contiguous().view(self.sample_size, -1)
        # Return -Inf if value is outside the support.
        if not selection_mask.any():
            return self._log_weights.new_zeros(torch.Size()).log()
        idxs = torch.arange(self.sample_size)[selection_mask.min(dim=-1)[0]]
        log_probs = self._categorical.log_prob(idxs)
        return log_sum_exp(log_probs)

    @property
    def batch_shape(self):
        return torch.Size()

    @property
    def event_shape(self):
        self._finalize()
        if self._samples is None:
            return None
        return self._samples.size()[1:]

    @property
    def mean(self):
        self._finalize()
        if self._samples.dtype in (torch.int32, torch.int64):
            raise ValueError("Mean for discrete empirical distribution undefined. Consider converting samples " +
                             "to ``torch.float32`` or ``torch.float64``.")
        weights = self._log_weights
        for _ in range(self._samples.dim() - 1):
            weights = weights.unsqueeze(-1)
        return (log_sum_exp(weights, scale=self._samples, dim=0) - log_sum_exp(weights, dim=0)).exp()

    @property
    def variance(self):
        self._finalize()
        if self._samples.dtype in (torch.int32, torch.int64):
            raise ValueError("Variance for discrete empirical distribution undefined. Consider converting samples " +
                             "to ``torch.float32`` or ``torch.float64``.")
        deviation_squared = torch.pow(self._samples - self.mean, 2)
        weights = self._log_weights
        for _ in range(self._samples.dim() - 1):
            weights = weights.unsqueeze(-1)
        return (log_sum_exp(weights, scale=deviation_squared, dim=0) - log_sum_exp(weights, dim=0)).exp()
