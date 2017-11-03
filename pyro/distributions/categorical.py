from __future__ import absolute_import, division, print_function

import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import get_probs_and_logits, torch_eye, torch_multinomial, torch_zeros_like


class Categorical(Distribution):
    """
    Categorical (discrete) distribution.

    Discrete distribution over elements of `vs` with :math:`P(vs[i]) \\propto ps[i]`.
    If ``one_hot=True``, ``.sample()`` returns a one-hot vector;
    otherwise ``.sample()`` returns the category index.

    :param ps: Probabilities. These should be non-negative and normalized
        along the rightmost axis.
    :type ps: torch.autograd.Variable
    :param logits: Log probability values. When exponentiated, these should
        sum to 1 along the last axis. Either `ps` or `logits` should be
        specified but not both.
    :type logits: torch.autograd.Variable
    :param vs: Optional list of values in the support.
    :type vs: list or numpy.ndarray or torch.autograd.Variable
    :param one_hot: Whether ``sample()`` returns a `one_hot` sample.  Defaults
        to `False` if `vs` is specified, or `True` if `vs` is not specified.
    :param batch_size: Optional number of elements in the batch used to
        generate a sample. The batch dimension will be the leftmost dimension
        in the generated sample.
    :type batch_size: int
    """
    enumerable = True

    def __init__(self, ps=None, vs=None, logits=None, one_hot=True, batch_size=None, log_pdf_mask=None, *args,
                 **kwargs):
        if (ps is None) == (logits is None):
            raise ValueError("Got ps={}, logits={}. Either `ps` or `logits` must be specified, "
                             "but not both.".format(ps, logits))
        self.ps, self.logits = get_probs_and_logits(ps=ps, logits=logits, is_multidimensional=True)
        # vs is None, Variable(Tensor), or numpy.array
        self.vs = self._process_data(vs)
        self.one_hot = one_hot
        self.log_pdf_mask = log_pdf_mask
        if vs is not None:
            vs_shape = self.vs.shape if isinstance(self.vs, np.ndarray) else self.vs.size()
            if vs_shape != ps.size():
                raise ValueError("Expected vs.size() or vs.shape == ps.size(), but got {} vs {}"
                                 .format(vs_shape, ps.size()))
            if self.one_hot:
                self.one_hot = False
        if self.ps.dim() == 1 and batch_size is not None:
            self.ps = self.ps.expand(batch_size, self.ps.size(0))
            self.logits = self.logits.expand(batch_size, self.logits.size(0))
            if log_pdf_mask is not None and log_pdf_mask.dim() == 1:
                self.log_pdf_mask = log_pdf_mask.expand(batch_size, log_pdf_mask.size(0))
        super(Categorical, self).__init__(*args, **kwargs)

    def _process_data(self, x):
        if x is not None:
            if isinstance(x, list):
                x = np.array(x)
            elif not isinstance(x, (Variable, torch.Tensor, np.ndarray)):
                raise TypeError(("Data should be of type: list, Variable, Tensor, or numpy array"
                                 "but was of {}".format(str(type(x)))))
        return x

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`
        """
        event_dim = 1
        ps = self.ps
        if x is not None:
            x = self._process_data(x)
            x_shape = x.shape if isinstance(x, np.ndarray) else x.size()
            try:
                ps = self.ps.expand(x_shape[:-event_dim] + self.event_shape())
            except RuntimeError as e:
                raise ValueError("Parameter `ps` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(ps.size(), x.size(), str(e)))
        return ps.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`
        """
        event_dim = 1
        return self.ps.size()[-event_dim:]

    def shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.shape`
        """
        if self.one_hot:
            return self.batch_shape(x) + self.event_shape()
        return self.batch_shape(x) + (1,)

    def sample(self):
        """
        Returns a sample which has the same shape as `ps` (or `vs`), except
        that if ``one_hot=True`` (and no `vs` is specified), the last dimension
        will have the same size as the number of events. The type of the sample
        is `numpy.ndarray` if `vs` is a list or a numpy array, else a tensor
        is returned.

        :return: sample from the Categorical distribution
        :rtype: numpy.ndarray or torch.LongTensor
        """
        sample = torch_multinomial(self.ps.data, 1, replacement=True).expand(*self.shape())
        sample_one_hot = torch_zeros_like(self.ps.data).scatter_(-1, sample, 1)

        if self.vs is not None:
            if isinstance(self.vs, np.ndarray):
                sample_bool_index = sample_one_hot.cpu().numpy().astype(bool)
                return self.vs[sample_bool_index].reshape(*self.shape())
            else:
                return self.vs.masked_select(sample_one_hot.byte())
        if self.one_hot:
            return Variable(sample_one_hot)
        return Variable(sample)

    def batch_log_pdf(self, x):
        """
        Evaluates log probability densities for one or a batch of samples and parameters.
        The last dimension for `ps` encodes the event probabilities, and the remaining
        dimensions are considered batch dimensions.

        `ps` and `vs` are first broadcasted to the size of the data `x`. The
        data tensor is used to to create a mask over `vs` where a 1 in the mask
        indicates that the corresponding value in `vs` was selected. Since, `ps`
        and `vs` have the same size, this mask when applied over `ps` gives
        the probabilities of the selected events. The method returns the logarithm
        of these probabilities.

        :return: tensor with log probabilities for each of the batches.
        :rtype: torch.autograd.Variable
        """
        logits = self.logits
        vs = self.vs
        x = self._process_data(x)
        batch_pdf_shape = self.batch_shape(x) + (1,)
        # probability tensor mask when data is numpy
        if isinstance(x, np.ndarray):
            batch_vs_size = x.shape[:-1] + (vs.shape[-1],)
            vs = np.broadcast_to(vs, batch_vs_size)
            boolean_mask = torch.from_numpy((vs == x).astype(int))
        # probability tensor mask when data is pytorch tensor
        else:
            x = x.cuda() if logits.is_cuda else x.cpu()
            batch_ps_shape = self.batch_shape(x) + self.event_shape()
            logits = logits.expand(batch_ps_shape)

            if vs is not None:
                vs = vs.expand(batch_ps_shape)
                boolean_mask = (vs == x)
            elif self.one_hot:
                boolean_mask = x
            else:
                boolean_mask = torch_zeros_like(logits.data).scatter_(-1, x.data.long(), 1)
        boolean_mask = boolean_mask.cuda() if logits.is_cuda else boolean_mask.cpu()
        if not isinstance(boolean_mask, Variable):
            boolean_mask = Variable(boolean_mask)
        # apply log function to masked probability tensor
        batch_log_pdf = logits.masked_select(boolean_mask.byte()).contiguous().view(batch_pdf_shape)
        if self.log_pdf_mask is not None:
            batch_log_pdf = batch_log_pdf * self.log_pdf_mask
        return batch_log_pdf

    def enumerate_support(self):
        """
        Returns the categorical distribution's support, as a tensor along the first dimension.

        Note that this returns support values of all the batched RVs in lock-step, rather
        than the full cartesian product. To iterate over the cartesian product, you must
        construct univariate Categoricals and use itertools.product() over all univariate
        variables (but this is very expensive).

        :param ps: Tensor where the last dimension denotes the event probabilities, *p_k*,
            which must sum to 1. The remaining dimensions are considered batch dimensions.
        :type ps: torch.autograd.Variable
        :param vs: Optional parameter, enumerating the items in the support. This could either
            have a numeric or string type. This should have the same dimension as ``ps``.
        :type vs: list or numpy.ndarray or torch.autograd.Variable
        :param one_hot: Denotes whether one hot encoding is enabled. This is True by default.
            When set to false, and no explicit `vs` is provided, the last dimension gives
            the one-hot encoded value from the support.
        :type one_hot: boolean
        :return: Torch variable or numpy array enumerating the support of the categorical distribution.
            Each item in the return value, when enumerated along the first dimensions, yields a
            value from the distribution's support which has the same dimension as would be returned by
            sample. If ``one_hot=True``, the last dimension is used for the one-hot encoding.
        :rtype: torch.autograd.Variable or numpy.ndarray.
        """
        sample_shape = self.batch_shape() + (1,)
        support_samples_size = (self.event_shape()) + sample_shape
        vs = self.vs

        if vs is not None:
            if isinstance(vs, np.ndarray):
                return vs.transpose().reshape(*support_samples_size)
            else:
                return torch.transpose(vs, 0, -1).contiguous().view(support_samples_size)
        if self.one_hot:
            return Variable(torch.stack([t.expand_as(self.ps) for t in torch_eye(*self.event_shape())]))
        else:
            LongTensor = torch.cuda.LongTensor if self.ps.is_cuda else torch.LongTensor
            return Variable(
                torch.stack([LongTensor([t]).expand(sample_shape)
                             for t in torch.arange(0, *self.event_shape()).long()]))
