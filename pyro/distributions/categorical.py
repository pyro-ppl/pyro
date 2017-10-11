import numpy as np
import torch
from torch.autograd import Variable

from pyro.distributions.distribution import Distribution
from pyro.util import to_one_hot


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

    def _process_v(self, vs):
        if vs is not None:
            if isinstance(vs, tuple):
                # recursively turn tuples into lists
                vs = [list(x) for x in vs]
            if isinstance(vs, list):
                vs = np.array(vs)
            elif not isinstance(vs, (Variable, torch.Tensor, np.ndarray)):
                raise TypeError(("vs should be of type: list, Variable, Tensor, tuple, or numpy array"
                                 "but was of {}".format(str(type(vs)))))
        return vs

    def _process_p(self, ps, vs, batch_size=1):
        if ps is not None:
            if ps.dim() == 1:
                ps = ps.unsqueeze(0)
                if isinstance(vs, Variable):
                    vs = vs.unsqueeze(0)
            elif batch_size > 1:
                ps = ps.expand(batch_size, ps.size(0))
                if isinstance(vs, Variable):
                    vs = vs.expand(batch_size, vs.size(0))
        return ps, vs

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
        vs = self._process_v(vs)
        self.ps, self.vs = self._process_p(ps, vs, batch_size)
        self.one_hot = one_hot
        super(Categorical, self).__init__(batch_size=1, *args, **kwargs)

    def sample(self, ps=None, vs=None, one_hot=True, *args, **kwargs):
        ps, vs, one_hot = self._sanitize_input(ps, vs, one_hot)
        vs = self._process_v(vs)
        ps, vs = self._process_p(ps, vs)
        sample = Variable(torch.multinomial(ps.data, 1, replacement=True).type_as(ps.data))
        if vs is not None:
            if isinstance(vs, np.ndarray):
                # always returns a 2-d (unsqueezed 1-d) list
                if vs.ndim == 1:
                    vs = np.expand_dims(vs, axis=0)
                r = np.arange(vs.shape[0])
                # if vs.shape[0] == 1:
                #     return vs[r, sample.squeeze().data.numpy().astype("int")][0]
                # else:
                return vs[r, sample.squeeze().data.numpy().astype("int")].tolist()
            # vs is a torch.Tensor
            return torch.gather(vs, 1, sample.long())
        if one_hot:
            # convert to onehot vector
            return to_one_hot(sample, ps)
        return sample

    def batch_log_pdf(self, x, ps=None, vs=None, one_hot=True, batch_size=1, *args, **kwargs):
        ps, vs, one_hot = self._sanitize_input(ps, vs, one_hot)
        vs = self._process_v(vs)
        ps, vs = self._process_p(ps, vs, batch_size)
        if not isinstance(x, list):
            if x.dim() == 1 and batch_size == 1:
                x = x.unsqueeze(0)
            if ps.size(0) != x.size(0):
                # convert to int to one-hot
                ps = ps.expand_as(x)
            else:
                ps = ps
        if vs is not None:
            # get index of tensor
            if isinstance(vs, np.ndarray):
                # x is a list if vs was a list or np.array
                bm = torch.Tensor((vs == np.array(x)).tolist())
                ix = len(vs.shape) - 1
                x = torch.nonzero(bm).select(ix, ix)
            else:
                # x is a Variable(Tensor) as are ps and vs
                ix = vs.dim() - 1
                x = torch.nonzero(vs.eq(x.expand_as(vs)).data).select(ix, ix)
            x = Variable(x).unsqueeze(1)
        elif one_hot:
            return torch.sum(x * torch.log(ps), 1)
        return torch.log(torch.gather(ps, 1, x.long()))

    def log_pdf(self, x, ps=None, vs=None, one_hot=True, batch_size=1, *args, **kwargs):
        return torch.sum(self.batch_log_pdf(x, ps, vs, one_hot, batch_size, *args, **kwargs))

    def support(self, ps=None, vs=None, one_hot=True, *args, **kwargs):
        """
        Returns the categorical distribution's support, as a tensor along the first dimension.

        :param ps: numpy.ndarray where the last dimension denotes the event probabilities, *p_k*,
            whichmust sum to 1. The remaining dimensions are considered batch dimensions.
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
        vs = self._process_v(vs)
        event_size = ps.size()[-1]
        batch_size = ps.size()[:-1]

        if vs is not None:
            if not batch_size:
                return vs
            if isinstance(vs, np.ndarray):
                return vs.transpose()
            else:
                return torch.t(vs)
        if one_hot:
            if not batch_size:
                return Variable(torch.stack([t.expand_as(ps) for t in torch.eye(event_size).long()]))
            return Variable(torch.stack([t.expand_as(ps) for t in torch.eye(event_size).long()]))
        else:
            if not batch_size:
                return Variable(torch.arange(0, event_size)).long()
            return Variable(torch.stack([torch.LongTensor([t]).expand(*batch_size)
                                         for t in torch.arange(0, event_size).long()]))
