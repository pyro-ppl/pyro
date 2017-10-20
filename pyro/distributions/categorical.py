import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution


class Categorical(Distribution):
    """
    :param ps: probabilities (can be unnormalized) *(vector or real array [0,
               Infinity))*
    :param vs: support *(any numpy array, Variable, or python list)*
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

    def batch_shape(self, ps=None, vs=None, one_hot=True, *args, **kwargs):
        ps, vs, one_hot = self._sanitize_input(ps, vs, one_hot)
        return ps.size()[:-1]

    def event_shape(self, ps=None, vs=None, one_hot=True, *args, **kwargs):
        ps, vs, one_hot = self._sanitize_input(ps, vs, one_hot)
        if one_hot:
            return ps.size()[-1:]
        else:
            return torch.Size((1,))

    def sample(self, ps=None, vs=None, one_hot=True, *args, **kwargs):
        """
        Returns a sample which has the same shape as ``ps`` (or ``vs``), except
        that if ``one_hot=True`` (and no ``vs`` is specified), the last dimension
        will have the same size as the number of events. The type of the sample
        is numpy.ndarray if vs is a list or a numpy array, else a tensor is returned.

        :return: sample from the Categorical distribution
        :rtype: numpy.ndarray or torch.LongTensor
        """
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
                return vs.masked_select(sample_one_hot.byte())
        if one_hot:
            return Variable(sample_one_hot)
        return Variable(sample)

    def batch_log_pdf(self, x, ps=None, vs=None, one_hot=True, *args, **kwargs):
        """
        Evaluates log probability densities for one or a batch of samples and parameters.
        The last dimension for ``ps`` encodes the event probabilities, and the remaining
        dimensions are considered batch dimensions.

        ``ps`` and ``vs`` are first broadcasted to the size of the data ``x``. The
        data tensor is used to to create a mask over ``vs`` where a 1 in the mask
        indicates that the corresponding value in ``vs`` was selected. Since, ``ps``
        and ``vs`` have the same size, this mask when applied over ``ps`` gives
        the probabilities of the selected events. The method returns the logarithm
        of these probabilities.

        :return: tensor with log probabilities for each of the batches.
        :rtype: torch.autograd.Variable
        """
        ps, vs, one_hot = self._sanitize_input(ps, vs, one_hot)
        vs = self._process_vs(vs)
        if isinstance(x, list):
            x = np.array(x)
        # probability tensor mask when data is numpy
        if isinstance(x, np.ndarray):
            batch_pdf_size = x.shape[:-1] + (1,)
            batch_vs_size = x.shape[:-1] + (vs.shape[-1],)
            vs = np.broadcast_to(vs, batch_vs_size)
            boolean_mask = torch.from_numpy((vs == x).astype(int))
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
        support_samples_size = (event_size,) + sample_size

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
