import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import to_one_hot


class Categorical(Distribution):
    """
    :param ps: probabilities (can be unnormalized) *(vector or real array [0,
               Infinity))*
    :param vs: support *(any numpy array, Variable, or list)*
    :param one_hot: if ``True``, ``sample()`` returns a one_hot sample. ``True`` by default.

    Discrete distribution over elements of ``vs`` with ``P(vs[i])`` proportional to
    ``ps[i]``.  If ``one_hot=True``, ``sample`` returns a one-hot vector.
    Else, ``sample`` returns the category selected.
    """
    enumerable = True

    def _sanitize_input(self, ps, vs, one_hot):
        if ps is not None:
            # stateless distribution
            return ps, vs, one_hot
        elif self.ps is not None:
            # stateful distribution
            return self.ps, self.vs, self.one_hot
        else:
            raise ValueError("Parameter(s) were None")

    def _process_vs(self, vs):
        if vs is not None:
            if isinstance(vs, list):
                vs = np.array(vs)
            elif not isinstance(vs, (Variable, torch.Tensor, np.ndarray)):
                raise TypeError(("vs should be of type: list, Variable, Tensor, tuple, or numpy array"
                                 "but was of {}".format(str(type(vs)))))
        return vs

    def __init__(self, ps=None, vs=None, one_hot=True, batch_size=1, *args, **kwargs):
        """
        Instantiates a discrete distribution.
        Params:
          vs - tuple, list, numpy array, Variable, or torch tensor of values
          ps - torch tensor of probabilities (must be same size as `vs`)
          one_hot - return one-hot samples (when `vs` is None)
          batch_size - expand ps and vs by a batch dimension
        """
        self.ps = ps
        # vs is None, Variable(Tensor), or numpy.array
        self.vs = self._process_vs(vs)
        self.one_hot = one_hot
        super(Categorical, self).__init__(batch_size=1, *args, **kwargs)

    def sample(self, ps=None, vs=None, one_hot=True, *args, **kwargs):
        ps, vs, one_hot = self._sanitize_input(ps, vs, one_hot)
        vs = self._process_vs(vs)
        sample_size = ps.size()[:-1] + (1,)
        sample = torch.multinomial(ps.data, 1, replacement=True).expand(*sample_size)
        sample_one_hot = torch.zeros(ps.size()).scatter_(-1, sample, 1)

        if vs is not None:
            if isinstance(vs, np.ndarray):
                sample_bool_index = sample_one_hot.cpu().numpy().astype(bool)
                return vs[sample_bool_index].reshape(*sample_size)
            else:
                return vs.masked_select(sample_one_hot.byte()).expand(*sample_size)
        if one_hot:
            return Variable(sample_one_hot)
        else:
            return Variable(sample)

    def batch_log_pdf(self, x, ps=None, vs=None, one_hot=True, *args, **kwargs):
        ps, vs, one_hot = self._sanitize_input(ps, vs, one_hot)
        vs = self._process_vs(vs)
        if isinstance(x, list):
            x = np.array(x)
        # probability tensor mask when data is numpy
        if isinstance(x, np.ndarray):
            sample_size = x.shape[:-1] + (1,)
            batch_vs_size = x.shape[:-1] + (vs.shape[-1],)
            vs = np.broadcast_to(vs, batch_vs_size)
            boolean_mask = torch.Tensor((vs == x).astype(int))
        # probability tensor mask when data is pytorch tensor
        else:
            batch_pdf_size = x.size()[:-1] + (1,)
            batch_ps_size = x.size()[:-1] + (ps.size()[-1],)
            ps = ps.expand(*batch_ps_size)
            if vs is not None:
                vs = vs.expand(*batch_ps_size)
                boolean_mask = vs == x
            elif one_hot:
                boolean_mask = x
            else:
                boolean_mask = torch.zeros(ps.size()).scatter_(-1, x.data.long(), 1)
        # apply log function to masked probability tensor
        return torch.log(ps.masked_select(boolean_mask.byte()).contiguous().view(*batch_pdf_size))

    def support(self, ps=None, vs=None, one_hot=True, *args, **kwargs):
        """
        Returns the categorical distribution's support, as a tensor along the first dimension.

        Note that this returns support values of all the batched RVs in lock-step, rather
        than the full cartesian product. To iterate over the cartesian product, you must
        construct univariate Categoricals and use itertools.product() over all univariate
        variables (but this is very expensive).

        :param ps: numpy.ndarray where the last dimension denotes the event probabilities, *p_k*,
            which must sum to 1. The remaining dimensions are considered batch dimensions.
        :param vs: Optional parameter, enumerating the items in the support. This could either
            have a numeric or string type. This should have the same dimension as ``ps``.
        :param one_hot: Denotes whether one hot encoding is enabled. This is True by default.
            When set to false, and no explicit ``vs`` is provided, the last dimension gives
            the one-hot encoded value from the support.
        :return: torch variable or numpy array enumerating the support of the categorical distribution.
            Each item in the return value, when enumerated along the first dimensions, yields a
            value from the distribution's support which has the same dimension as would be returned by
            sample. If ``one_hot=True``, the last dimension is used for the one-hot encoding.
        :rtype: torch.autograd.Variable or numpy.ndarray.
        """
        ps, vs, one_hot = self._sanitize_input(ps, vs, one_hot)
        vs = self._process_vs(vs)
        event_size = ps.size()[-1]
        sample_size = ps.size()[:-1] + (1,)
        support_samples_size = (event_size, ) + sample_size

        if vs is not None:
            if isinstance(vs, np.ndarray):
                return vs.transpose().reshape(*support_samples_size)
            else:
                return torch.transpose(vs, 0, -1).contiguous().view(*support_samples_size)
        if one_hot:
            return Variable(torch.stack([t.expand_as(ps) for t in torch.eye(event_size)]))
        else:
            return Variable(torch.stack([torch.LongTensor([t]).expand(*sample_size)
                                         for t in torch.arange(0, event_size).long()]))
