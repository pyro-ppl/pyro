import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution
import itertools
import numpy as np
from pyro.util import to_one_hot


class Categorical(Distribution):
    """
    Categorical is a specialized version of multinomial where n = 1
    """

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
                ps = ps.unsqueeze(0).expand(batch_size, 0)
                if isinstance(vs, Variable):
                    vs = vs.unsqueeze(0).expand(batch_size, 0)
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
        _ps, _vs, _one_hot = self._sanitize_input(ps, vs, one_hot)
        _vs = self._process_v(_vs)
        _ps, _vs = self._process_p(_ps, _vs)
        sample = Variable(torch.multinomial(_ps.data, 1, replacement=True))
        if _vs is not None:
            if isinstance(_vs, np.ndarray):
                # always returns a 2-d (unsqueezed 1-d) list
                r = np.arange(_vs.shape[0])
                return [[x] for x in _vs[r, sample.squeeze().data.numpy()].tolist()]
            # _vs is a torch.Tensor
            return torch.gather(_vs, 1, sample.long())
        if _one_hot:
            # convert to onehot vector
            return to_one_hot(sample, _ps)
        return sample

    def batch_log_pdf(self, x, ps=None, vs=None, one_hot=True, batch_size=1, *args, **kwargs):
        _ps, _vs, _one_hot = self._sanitize_input(ps, vs, one_hot)
        _vs = self._process_v(_vs)
        _ps, _vs = self._process_p(_ps, _vs, batch_size)
        if not isinstance(x, list):
            if x.dim() == 1 and batch_size == 1:
                x = x.unsqueeze(0)
            if _ps.size(0) != x.size(0):
                # convert to int to one-hot
                _ps = _ps.expand_as(x)
            else:
                _ps = _ps
        if _vs is not None:
            # get index of tensor
            if isinstance(_vs, np.ndarray):
                # x is a list if _vs was a list or np.array
                bm = torch.Tensor((_vs == np.array(x)).tolist())
                ix = len(_vs.shape) - 1
                _x = torch.nonzero(bm).select(ix, ix)
            else:
                # x is a Variable(Tensor) as are ps and vs
                ix = _vs.dim() - 1
                _x = torch.nonzero(_vs.eq(x.expand_as(_vs)).data).select(ix, ix)
            x = Variable(_x).unsqueeze(1)
        elif _one_hot:
            return torch.sum(x * torch.log(_ps), 1)
        return torch.log(torch.gather(_ps, 1, x.long()))

    def log_pdf(self, x, ps=None, vs=None, one_hot=True, batch_size=1, *args, **kwargs):
        return torch.sum(self.batch_log_pdf(x, ps, vs, one_hot, batch_size, *args, **kwargs))

    def support(self, ps=None, vs=None, one_hot=True, *args, **kwargs):
        _ps, _vs, _one_hot = self._sanitize_input(ps, vs, one_hot)
        r = _ps.size(0)
        c = _ps.size(1)
        _vs = self._process_v(_vs)
        _ps, _vs = self._process_p(_ps, _vs)

        if _vs is not None:
            if isinstance(_vs, np.ndarray):
                # vs is an array, so the support must be of type array
                r_np = _vs.shape[0]
                c_np = _vs.shape[1]
                ix = np.expand_dims(np.arange(r_np), axis=1)
                b = torch.ones(r_np, 1)
                return (_vs[np.arange(r_np), torch.Tensor(list(x)).numpy().astype(int)]
                        .reshape(r_np, 1).tolist()
                        for x in itertools.product(torch.arange(0, c_np), repeat=r_np))
            # vs is a tensor so support is of type tensor
            return (torch.sum(_vs * Variable(torch.Tensor(list(x))), 1)
                    for x in itertools.product(torch.eye(c).numpy().tolist(),
                    repeat=r))

        if _one_hot:
            return (Variable(torch.Tensor(list(x)))
                    for x in itertools.product(torch.eye(c).numpy().tolist(),
                    repeat=r))

        if r == 1:
            return (Variable(torch.Tensor([[i]])) for i in range(c))
        return (Variable(torch.Tensor(list(x)).unsqueeze(1))
                for x in itertools.product(torch.arange(0, c),
                repeat=r))
