from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import copy_docs_from, get_probs_and_logits, torch_eye, torch_multinomial, torch_zeros_like


@copy_docs_from(Distribution)
class OneHotCategorical(Distribution):
    """
    OneHotCategorical (discrete) distribution.

    Discrete distribution over one-hot vectors.

    :param ps: Probabilities. These should be non-negative and normalized
        along the rightmost axis.
    :type ps: torch.autograd.Variable
    :param logits: Log probability values. When exponentiated, these should
        sum to 1 along the last axis. Either `ps` or `logits` should be
        specified but not both.
    :type logits: torch.autograd.Variable
    :param batch_size: Optional number of elements in the batch used to
        generate a sample. The batch dimension will be the leftmost dimension
        in the generated sample.
    :type batch_size: int
    """
    enumerable = True

    def __init__(self, ps=None, logits=None, batch_size=None, log_pdf_mask=None, *args, **kwargs):
        if (ps is None) == (logits is None):
            raise ValueError("Got ps={}, logits={}. Either `ps` or `logits` must be specified, "
                             "but not both.".format(ps, logits))
        self.ps, self.logits = get_probs_and_logits(ps=ps, logits=logits, is_multidimensional=True)
        self.log_pdf_mask = log_pdf_mask
        if batch_size is not None:
            if self.ps.dim() != 1:
                raise NotImplementedError
            self.ps = self.ps.expand(batch_size, *self.ps.size())
            self.logits = self.logits.expand(batch_size, *self.logits.size())
            if log_pdf_mask is not None:
                self.log_pdf_mask = log_pdf_mask.expand(batch_size, *log_pdf_mask.size())
        super(OneHotCategorical, self).__init__(*args, **kwargs)

    def _process_data(self, x):
        if x is not None and not isinstance(x, (Variable, torch.Tensor)):
            raise TypeError("Expected data of type Variable or Tensor, got {}".format(type(x)))
        return x

    def batch_shape(self, x=None):
        event_dim = 1
        ps = self.ps
        if x is not None:
            x = self._process_data(x)
            try:
                ps = self.ps.expand(x.size()[:-event_dim] + self.event_shape())
            except RuntimeError as e:
                raise ValueError("Parameter `ps` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(ps.size(), x.size(), e))
        return ps.size()[:-event_dim]

    def event_shape(self):
        event_dim = 1
        return self.ps.size()[-event_dim:]

    def shape(self, x=None):
        return self.batch_shape(x) + self.event_shape()

    def sample(self):
        """
        Returns a sample which has the same shape as `ps`, except that the last dimension
        will have the same size as the number of events.

        :return: sample from the OneHotCategorical distribution
        :rtype: torch.Tensor
        """
        sample = torch_multinomial(self.ps.data, 1, replacement=True).expand(*self.shape())
        sample_one_hot = torch_zeros_like(self.ps.data).scatter_(-1, sample, 1)
        return Variable(sample_one_hot)

    def batch_log_pdf(self, x):
        """
        Evaluates log probability densities for one or a batch of samples and
        parameters.  The last dimension for `ps` encodes the event
        probabilities, and the remaining dimensions are considered batch
        dimensions.

        `ps` and first broadcasted to the size of the data `x`. The data tensor
        is used to to create a mask over `ps` where a 1 in the mask indicates
        that the corresponding probability in `ps` was selected. The method
        returns the logarithm of these probabilities.

        :return: tensor with log probabilities for each of the batches.
        :rtype: torch.autograd.Variable
        """
        logits = self.logits
        x = self._process_data(x)
        if not isinstance(x, Variable):
            x = Variable(x)
        batch_pdf_shape = self.batch_shape(x) + (1,)
        batch_ps_shape = self.batch_shape(x) + self.event_shape()
        logits = logits.expand(batch_ps_shape)
        boolean_mask = x.cuda(logits.get_device()) if logits.is_cuda else x.cpu()
        # apply log function to masked probability tensor
        batch_log_pdf = logits.masked_select(boolean_mask.byte()).contiguous().view(batch_pdf_shape)
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf

    def enumerate_support(self):
        """
        Returns the categorical distribution's support, as a tensor along the
        first dimension.

        Note that this returns support values of all the batched RVs in
        lock-step, rather than the full cartesian product. To iterate over the
        cartesian product, you must construct univariate Categoricals and use
        itertools.product() over all univariate variables (but this is very
        expensive).

        :param torch.autograd.Variable ps: Tensor where the last dimension
            denotes the event probabilities, *p_k*, which must sum to 1. The
            remaining dimensions are considered batch dimensions.
        :return: Torch variable enumerating the support of the categorical
            distribution.  Each item in the return value, when enumerated along
            the first dimensions, yields a value from the distribution's
            support which has the same dimension as would be returned by
            sample. The last dimension is used for the one-hot encoding.
        :rtype: torch.autograd.Variable.
        """
        result = torch.stack([t.expand_as(self.ps) for t in torch_eye(*self.event_shape())])
        if self.ps.is_cuda:
            result = result.cuda(self.ps.get_device())
        return Variable(result)

    def analytic_mean(self):
        return self.ps

    def analytic_var(self):
        return self.ps * (1 - self.ps)
