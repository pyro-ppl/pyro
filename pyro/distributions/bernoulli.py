from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import get_probs_and_logits


class Bernoulli(Distribution):
    """
    Bernoulli distribution.

    Distribution over a vector of independent Bernoulli variables. Each element
    of the vector takes on a value in `{0, 1}`.

    This is often used in conjunction with `torch.nn.Sigmoid` to ensure the
    `ps` parameters are in the interval `[0, 1]`.

    :param torch.autograd.Variable ps: Probabilities. Should lie in the
        interval `[0,1]`.
    :param logits: Log odds, i.e. :math:`\\log(\\frac{p}{1 - p})`. Either `ps` or
        `logits` should be specified, but not both.
    :param batch_size: The number of elements in the batch used to generate
        a sample. The batch dimension will be the leftmost dimension in the
        generated sample.
    :param log_pdf_mask: Tensor that is applied to the batch log pdf values
        as a multiplier. The most common use case is supplying a boolean
        tensor mask to mask out certain batch sites in the log pdf computation.
    """
    enumerable = True

    def __init__(self, ps=None, logits=None, batch_size=None, log_pdf_mask=None, *args, **kwargs):
        if (ps is None) == (logits is None):
            raise ValueError("Got ps={}, logits={}. Either `ps` or `logits` must be specified, "
                             "but not both.".format(ps, logits))
        self.ps, self.logits = get_probs_and_logits(ps=ps, logits=logits, is_multidimensional=False)
        self.log_pdf_mask = log_pdf_mask
        if self.ps.dim() == 1 and batch_size is not None:
            self.ps = self.ps.expand(batch_size, self.ps.size(0))
            self.logits = self.logits.expand(batch_size, self.logits.size(0))
            if log_pdf_mask is not None and log_pdf_mask.dim() == 1:
                self.log_pdf_mask = log_pdf_mask.expand(batch_size, log_pdf_mask.size(0))
        super(Bernoulli, self).__init__(*args, **kwargs)

    def batch_shape(self, x=None):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_shape`.
        """
        event_dim = 1
        ps = self.ps
        if x is not None:
            if x.size()[-event_dim] != ps.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.ps.size()[-1], but got {} vs {}".format(
                                     x.size(-1), ps.size(-1)))
            try:
                ps = self.ps.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `ps` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(ps.size(), x.size(), str(e)))

        return ps.size()[:-event_dim]

    def event_shape(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.event_shape`.
        """
        event_dim = 1
        return self.ps.size()[-event_dim:]

    def sample(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`.
        """
        return Variable(torch.bernoulli(self.ps.data))

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        max_val = (-self.logits).clamp(min=0)
        binary_cross_entropy = self.logits - self.logits * x + max_val + \
            ((-max_val).exp() + (-self.logits - max_val).exp()).log()
        log_prob = -binary_cross_entropy
        # XXX this allows for the user to mask out certain parts of the score, for example
        # when the data is a ragged tensor. also useful for KL annealing. this entire logic
        # will likely be done in a better/cleaner way in the future
        if self.log_pdf_mask is not None:
            log_prob = log_prob * self.log_pdf_mask
        return torch.sum(log_prob, -1).contiguous().view(batch_log_pdf_shape)

    def enumerate_support(self):
        """
        Returns the Bernoulli distribution's support, as a tensor along the first dimension.

        Note that this returns support values of all the batched RVs in lock-step, rather
        than the full cartesian product. To iterate over the cartesian product, you must
        construct univariate Bernoullis and use itertools.product() over all univariate
        variables (may be expensive).

        :return: torch variable enumerating the support of the Bernoulli distribution.
            Each item in the return value, when enumerated along the first dimensions, yields a
            value from the distribution's support which has the same dimension as would be returned by
            sample.
        :rtype: torch.autograd.Variable.
        """
        return Variable(torch.stack([torch.Tensor([t]).expand_as(self.ps) for t in [0, 1]]))

    def analytic_mean(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_mean`.
        """
        return self.ps

    def analytic_var(self):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.analytic_var`.
        """
        return self.ps * (1 - self.ps)
