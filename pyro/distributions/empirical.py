from __future__ import absolute_import, division, print_function

import math
import numbers
from collections import defaultdict

import torch
from contextlib2 import contextmanager
from torch.distributions import constraints

from pyro.distributions.torch import Categorical
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import copy_docs_from, logsumexp


@contextmanager
def accumulate_samples():
    empirical_dist = Empirical()
    yield empirical_dist
    empirical_dist._finalize()


@copy_docs_from(TorchDistribution)
class Empirical(TorchDistribution):
    r"""
    Empirical distribution associated with the sampled data.
    """

    arg_constraints = {}
    support = constraints.real
    has_enumerate_support = True

    def __init__(self, validate_args=None):
        self._samples = None
        self._log_weights = None
        self._categorical = None
        self._samples_buffer = defaultdict(list)
        self._weights_buffer = defaultdict(list)
        super(TorchDistribution, self).__init__(batch_shape=torch.Size(), validate_args=validate_args)

    def _finalize(self):
        """
        Appends values collected in the samples/weights buffers to their
        corresponding tensors.
        """
        num_chains = len(self._samples_buffer)
        samples_by_chain = []
        weights_by_chain = []
        for i in range(num_chains):
            samples_by_chain.append(torch.stack(self._samples_buffer[i], dim=0))
            weights_by_chain.append(torch.stack(self._weights_buffer[i], dim=0))
        if num_chains == 1:
            self._samples = samples_by_chain[0]
            self._log_weights = weights_by_chain[0]
        else:
            self._samples = torch.stack(samples_by_chain, dim=0)
            self._log_weights = torch.stack(weights_by_chain, dim=0)
        self._categorical = Categorical(logits=self._log_weights)

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

    def add(self, value, weight=None, log_weight=None, chain_id=0):
        """
        Adds a new data point to the sample. The values in successive calls to
        ``add`` must have the same tensor shape and size. Optionally, an
        importance weight can be specified via ``log_weight`` or ``weight``
        (default value of `1` is used if not specified).

        :param torch.Tensor value: tensor to add to the sample.
        :param torch.Tensor weight: log weight (optional) corresponding
            to the sample.
        :param torch.Tensor log_weight: weight (optional) corresponding
            to the sample.
        """
        if self._validate_args:
            if weight is not None and log_weight is not None:
                raise ValueError("Only one of ```weight`` or ``log_weight`` should be specified.")

        weight_type = value.new_empty(1).float().type() if value.dtype in (torch.int32, torch.int64) \
            else value.type()
        # Apply default weight of 1.0.
        if log_weight is None and weight is None:
            log_weight = torch.tensor(0.0).type(weight_type)
        elif weight is not None and log_weight is None:
            log_weight = math.log(weight)
        if isinstance(log_weight, numbers.Number):
            log_weight = torch.tensor(log_weight).type(weight_type)
        if self._validate_args and log_weight.dim() > 0:
            raise ValueError("``weight.dim() > 0``, but weight should be a scalar.")

        # Append to the buffer list
        self._samples_buffer[chain_id].append(value)
        self._weights_buffer[chain_id].append(log_weight)

    def sample(self, sample_shape=torch.Size()):
        self._finalize()
        idxs = self._categorical.sample(sample_shape=sample_shape)
        return self._samples[idxs]

    def log_prob(self, value):
        """
        Returns the log of the probability mass function evaluated at ``value``.
        Note that this currently only supports scoring values with empty
        ``sample_shape``, i.e. an arbitrary batched sample is not allowed.

        :param torch.Tensor value: scalar or tensor value to be scored.
        """
        if self._validate_args:
            if value.shape != self.event_shape:
                raise ValueError("``value.shape`` must be {}".format(self.event_shape))
        self._finalize()
        selection_mask = self._samples.eq(value).contiguous().view(self.sample_size, -1)
        # Return -Inf if value is outside the support.
        if not selection_mask.any():
            return self._log_weights.new_zeros(torch.Size()).log()
        idxs = torch.arange(self.sample_size)[selection_mask.min(dim=-1)[0]]
        log_probs = self._categorical.log_prob(idxs)
        return logsumexp(log_probs, dim=-1)

    def _weighted_mean(self, value, dim=0):
        weights = self._log_weights
        for _ in range(value.dim() - 1):
            weights = weights.unsqueeze(-1)
        max_val = weights.max(dim)[0]
        return max_val.exp() * (value * (weights - max_val.unsqueeze(-1)).exp()).sum(dim=dim)

    @property
    def event_shape(self):
        self._finalize()
        if self._samples is None:
            return None
        return self._samples.shape[1:]

    @property
    def mean(self):
        self._finalize()
        if self._samples.dtype in (torch.int32, torch.int64):
            raise ValueError("Mean for discrete empirical distribution undefined. " +
                             "Consider converting samples to ``torch.float32`` " +
                             "or ``torch.float64``. If these are samples from a " +
                             "`Categorical` distribution, consider converting to a " +
                             "`OneHotCategorical` distribution.")
        return self._weighted_mean(self._samples) / self._weighted_mean(self._samples.new_tensor([1.]))

    @property
    def variance(self):
        self._finalize()
        if self._samples.dtype in (torch.int32, torch.int64):
            raise ValueError("Variance for discrete empirical distribution undefined. " +
                             "Consider converting samples to ``torch.float32`` " +
                             "or ``torch.float64``. If these are samples from a " +
                             "`Categorical` distribution, consider converting to a " +
                             "`OneHotCategorical` distribution.")
        deviation_squared = torch.pow(self._samples - self.mean, 2)
        return self._weighted_mean(deviation_squared) / self._weighted_mean(self._samples.new_tensor([1.]))

    def get_samples_and_weights(self):
        self._finalize()
        return self._samples, self._log_weights

    def enumerate_support(self, expand=True):
        # Empirical does not support batching, so expanding is a no-op.
        self._finalize()
        return self._samples
