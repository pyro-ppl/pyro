from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.transformed_distribution import ExpBijector, TransformedDistribution
from pyro.distributions.util import _get_clamping_buffer, get_probs_and_logits, log_gamma


class ExpRelaxedCategorical(Distribution):
    """ExpRelaxedCategorical distribution.

    Conditinuous version of the log of a Categorical distribution using softmax
    relaxation of Gumbel-Max distribution. Returns a log of a point in the
    simplex. Based on the interface to OneHotCategorical

    Implementation based on [1].

    :param torch.Tensor temperature: Relaxation temperature. Should be a
        positive, 1-dimensional Tensor.
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

    [1] THE CONCRETE DISTRIBUTION: A CONTINUOUS RELAXATION OF DISCRETE RANDOM VARIABLES
    (Maddison et al, 2017)
    [2] CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX
    (Jang et al, 2017)
    """
    reparameterized = True

    def __init__(self, temperature, ps=None, logits=None, batch_size=None, log_pdf_mask=None, *args, **kwargs):
        self.temperature = temperature
        if (ps is None) == (logits is None):
            raise ValueError("Got ps={}, logits={}. Either `ps` or `logits` must be specified, "
                             "but not both.".format(ps, logits))
        self.ps, self.logits = get_probs_and_logits(ps, logits, is_multidimensional=True)
        self.log_pdf_mask = log_pdf_mask
        if self.ps.dim() == 1 and batch_size is not None:
            self.ps = self.ps.expand(batch_size, self.ps.size(0))
            self.logits = self.logits.expand(batch_size, self.logits.size(0))
            if log_pdf_mask is not None and log_pdf_mask.dim() == 1:
                self.log_pdf_mask = log_pdf_mask.expand(batch_size, log_pdf_mask.size(0))
        super(ExpRelaxedCategorical, self).__init__(*args, **kwargs)

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
            try:
                ps = self.ps.expand(x.size()[:-event_dim] + self.event_shape())
            except RuntimeError as e:
                raise ValueError("Parameter `ps` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(ps.size(), x.size(), e))
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
        return self.batch_shape(x) + self.event_shape()

    def sample(self):
        """
        Draws either a single sample (if self.logits.dim() == 1), or one sample per param (if self.logits.dim() == 2).
        Reparameterized.
        """

        # Sample Gumbels, G_k = -log(-log(U))
        uniforms = torch.zeros(self.logits.data.size()).uniform_()
        eps = _get_clamping_buffer(uniforms)
        uniforms = uniforms.clamp(min=eps, max=1 - eps)
        gumbels = Variable(uniforms.log().mul(-1).log().mul(-1))

        # Reparameterize
        z = F.log_softmax((self.logits + gumbels) / self.temperature, dim=-1)
        return z if self.reparameterized else z.detach()

    def batch_log_pdf(self, x):
        """
        Evaluates log probability densities for one or a batch of samples and parameters.

        :return: tensor with log probabilities for each of the batches.
        :rtype: torch.autograd.Variable
        """
        n = self.event_shape()[0]
        logits = self.logits.expand(self.shape(x))
        log_scale = Variable(log_gamma(
            torch.Tensor([n]).expand(self.batch_shape(x)))) - self.temperature.log().mul(-(n - 1))
        scores = logits.log() + x.mul(-self.temperature)
        scores = scores.sum(dim=-1)

        log_part = n * logits.mul(x.mul(-self.temperature).exp()).sum(dim=-1).log()
        return (scores - log_part + log_scale).contiguous().view(self.batch_shape(x) + (1,))


class RelaxedCategorical(Distribution):

    def __new__(cls, *args, **kwargs):
        return TransformedDistribution(ExpRelaxedCategorical(*args, **kwargs), ExpBijector())
