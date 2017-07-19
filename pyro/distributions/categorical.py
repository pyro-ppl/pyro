import torch
from torch.autograd import Variable
from pyro.distributions.distribution import Distribution
import itertools
import numpy as np


def _to_one_hot(x, ps):
    batch_size = x.size(0)
    classes = ps.size(1)
    # create an empty array for one-hots
    batch_one_hot = torch.zeros(batch_size, classes)
    # this operation writes ones where needed
    batch_one_hot.scatter_(1, x.data.view(-1, 1), 1)

    return Variable(batch_one_hot)


class Categorical(Distribution):
    """
    Categorical is a specialized version of multinomial where n = 1
    """

    def __init__(self, ps, vs=None, one_hot=True, batch_size=1, *args, **kwargs):
        """
        Instantiates a discrete distribution.
        Params:
          vs - tuple, list, numpy array, Variable, or torch tensor of values
          ps - torch tensor of probabilities (must be same size as `vs`)
          one_hot - return one-hot samples (when `vs` is None)
          batch_size - expand ps and vs by a batch dimension
        """
        if vs is not None:
            if isinstance(vs, tuple):
                # recursively turn tuples into lists
                vs = [list(x) for x in vs]
            if isinstance(vs, list):
                vs = np.array(vs)
            elif not isinstance(vs, (Variable, torch.Tensor, np.ndarray)):
                raise TypeError(("vs should be of type: list, Variable, Tensor, tuple, or numpy array"
                                 "but was of {}".format(str(type(vs)))))
        self.ps = ps
        # vs is None, Variable(Tensor), or numpy.array
        self.vs = vs
        if ps.dim() == 1:
            self.ps = ps.unsqueeze(0)
            if isinstance(vs, Variable):
                self.vs = vs.unsqueeze(0)
        elif batch_size > 1:
            self.ps = ps.unsqueeze(0).expand(batch_size, 0)
            if isinstance(vs, Variable):
                self.vs = vs.unsqueeze(0).expand(batch_size, 0)
        self.one_hot = one_hot
        super(Categorical, self).__init__(batch_size=1, *args, **kwargs)

    def sample(self):
        sample = Variable(torch.multinomial(self.ps.data, 1, replacement=True))
        if self.vs is not None:
            if isinstance(self.vs, np.ndarray):
                # always returns a 2-d (unsqueezed 1-d) list
                r = np.arange(self.vs.shape[0])
                return [[x] for x in self.vs[r, sample.squeeze().data.numpy()].tolist()]
            # self.vs is a torch.Tensor
            return torch.gather(self.vs, 1, sample.long())
        if self.one_hot:
            # convert to onehot vector
            return _to_one_hot(sample, self.ps)
        return sample

    def batch_log_pdf(self, x, batch_size=1):
        if not isinstance(x, list):
            if x.dim() == 1 and batch_size == 1:
                x = x.unsqueeze(0)
            if self.ps.size(0) != x.size(0):
                # convert to int to one-hot
                _ps = self.ps.expand_as(x)
            else:
                _ps = self.ps
        if self.vs is not None:
            # get index of tensor
            if isinstance(self.vs, np.ndarray):
                # x is a list if self.vs was a list or np.array
                bm = torch.Tensor((self.vs == np.array(x)).tolist())
                ix = len(self.vs.shape) - 1
                _x = torch.nonzero(bm).select(ix, ix)
            else:
                # x is a Variable(Tensor) as are ps and vs
                ix = self.vs.dim() - 1
                _x = torch.nonzero(self.vs.eq(x.expand_as(self.vs)).data).select(ix, ix)
            x = Variable(_x).unsqueeze(1)
        elif self.one_hot:
            return torch.sum(x * torch.log(_ps), 1)
        return torch.log(torch.gather(self.ps, 1, x.long()))

    def log_pdf(self, _x):
        return torch.sum(self.batch_log_pdf(_x))

    def support(self):
        r = self.ps.size(0)
        c = self.ps.size(1)

        if self.vs is not None:
            if isinstance(self.vs, np.ndarray):
                # vs is an array, so the support must be of type array
                r_np = self.vs.shape[0]
                c_np = self.vs.shape[1]
                ix = np.expand_dims(np.arange(r_np), axis=1)
                b = torch.ones(r_np, 1)
                return (self.vs[np.arange(r_np), torch.Tensor(list(x)).numpy().astype(int)]
                        .reshape(r_np, 1).tolist()
                        for x in itertools.product(torch.arange(0, c_np), repeat=r_np))
            # vs is a tensor so support is of type tensor
            return (torch.sum(self.vs * Variable(torch.Tensor(list(x))), 1)
                    for x in itertools.product(torch.eye(c).numpy().tolist(),
                    repeat=r))

        if self.one_hot:
            return (Variable(torch.Tensor(list(x)))
                    for x in itertools.product(torch.eye(c).numpy().tolist(),
                    repeat=r))

        if r == 1:
            return (Variable(torch.Tensor([[i]])) for i in range(c))
        return (Variable(torch.Tensor(list(x)).unsqueeze(1))
                for x in itertools.product(torch.arange(0, c),
                repeat=r))
