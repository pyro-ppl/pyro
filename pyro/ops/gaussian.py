import math

import torch
from torch.nn.functional import pad

from pyro.distributions.util import broadcast_shape


class Gaussian(object):
    """
    Non-normalized Gaussian distribution.

    This represents an arbitrary semidefinite quadratic function, which can be
    interpreted as a rank-deficient scaled Gaussian distribution. The precision
    matrix may have zero eigenvalues, thus it may be impossible to work
    directly with the covariance matrix.

    :param torch.Tensor log_normalizer: a normalization constant, which is mainly used to keep
        track of normalization terms during contractions.
    :param torch.Tensor info_vec: information vector, which is a scaled version of the mean
        ``info_vec = precision @ mean``. We use this represention to make gaussian contraction
        fast and stable.
    :param torch.Tensor precision: precision matrix of this gaussian.
    """
    def __init__(self, log_normalizer, info_vec, precision):
        # NB: using info_vec instead of mean to deal with rank-deficient problem
        assert info_vec.dim() >= 1
        assert precision.dim() >= 2
        assert precision.shape[-2:] == info_vec.shape[-1:] * 2
        self.log_normalizer = log_normalizer
        self.info_vec = info_vec
        self.precision = precision

    def dim(self):
        return self.info_vec.size(-1)

    @property
    def batch_shape(self):
        return broadcast_shape(self.log_normalizer.shape,
                               self.info_vec.shape[:-1],
                               self.precision.shape[:-2])

    def log_density(self, value):
        """
        Evaluate the log density of this Gaussian at a point value.

            `-0.5 * value.T @ precision @ value + value.T @ info_vec + log_normalizer`

        This is mainly used for testing.
        """
        if value.size(-1) == 0:
            batch_shape = broadcast_shape(value.shape[:-1], self.batch_shape)
            return self.log_normalizer.expand(batch_shape)
        result = (-0.5) * torch.matmul(self.precision, value.unsqueeze(-1)).squeeze(-1)
        result = result + self.info_vec
        result = (value * result).sum(-1)
        return result + self.log_normalizer


def gaussian_tensordot(x, y, dims=0):
    """
    Computes the integral over two gaussians:

        `(x @ y)(a,c) = log(integral(exp(x(a,b) + y(b,c)), b))`,

    where `x` is a gaussian over variables (a,b), `y` is a gaussian over variables
    (b,c), (a,b,c) can each be sets of zero or more variables, and `dims` is the size of b.

    :param x: a Gaussian instance
    :param y: a Gaussian instance
    :param dims: number of variables to contract
    """
    assert isinstance(x, Gaussian)
    assert isinstance(y, Gaussian)
    na = x.dim() - dims
    nb = dims
    nc = y.dim() - dims
    assert na >= 0
    assert nb >= 0
    assert nc >= 0

    Paa, Pba, Pbb = x.precision[..., :na, :na], x.precision[..., na:, :na], x.precision[..., na:, na:]
    Qbb, Qbc, Qcc = y.precision[..., :nb, :nb], y.precision[..., :nb, nb:], y.precision[..., nb:, nb:]
    xa, xb = x.info_vec[..., :na], x.info_vec[..., na:]  # x.precision @ x.mean
    yb, yc = y.info_vec[..., :nb], y.info_vec[..., nb:]  # y.precision @ y.mean

    precision = pad(Paa, (0, nc, 0, nc)) + pad(Qcc, (na, 0, na, 0))
    info_vec = pad(xa, (0, nc)) + pad(yc, (na, 0))
    log_normalizer = x.log_normalizer + y.log_normalizer
    if nb > 0:
        B = pad(Pba, (0, nc)) + pad(Qbc, (na, 0))
        b = xb + yb

        # Pbb + Qbb needs to be positive definite, so that we can malginalize out `b` (to have a finite integral)
        L = torch.cholesky(Pbb + Qbb)
        LinvB = torch.triangular_solve(B, L, upper=False)[0]
        LinvBt = LinvB.transpose(-2, -1)
        Linvb = torch.triangular_solve(b.unsqueeze(-1), L, upper=False)[0]

        precision = precision - torch.matmul(LinvBt, LinvB)
        # NB: precision might not be invertible for getting mean = precision^-1 @ info_vec
        if na + nc > 0:
            info_vec = info_vec - torch.matmul(LinvBt, Linvb).squeeze(-1)
        logdet = torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)
        diff = 0.5 * nb * math.log(2 * math.pi) + 0.5 * Linvb.squeeze(-1).pow(2).sum(-1) - logdet
        log_normalizer = log_normalizer + diff

    return Gaussian(log_normalizer, info_vec, precision)
