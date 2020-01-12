# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions.utils import lazy_property
from torch.nn.functional import pad

from pyro.distributions.util import broadcast_shape


class Gaussian:
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

    @lazy_property
    def batch_shape(self):
        return broadcast_shape(self.log_normalizer.shape,
                               self.info_vec.shape[:-1],
                               self.precision.shape[:-2])

    def expand(self, batch_shape):
        n = self.dim()
        log_normalizer = self.log_normalizer.expand(batch_shape)
        info_vec = self.info_vec.expand(batch_shape + (n,))
        precision = self.precision.expand(batch_shape + (n, n))
        return Gaussian(log_normalizer, info_vec, precision)

    def reshape(self, batch_shape):
        n = self.dim()
        log_normalizer = self.log_normalizer.reshape(batch_shape)
        info_vec = self.info_vec.reshape(batch_shape + (n,))
        precision = self.precision.reshape(batch_shape + (n, n))
        return Gaussian(log_normalizer, info_vec, precision)

    def __getitem__(self, index):
        """
        Index into the batch_shape of a Gaussian.
        """
        assert isinstance(index, tuple)
        log_normalizer = self.log_normalizer[index]
        info_vec = self.info_vec[index + (slice(None),)]
        precision = self.precision[index + (slice(None), slice(None))]
        return Gaussian(log_normalizer, info_vec, precision)

    @staticmethod
    def cat(parts, dim=0):
        """
        Concatenate a list of Gaussians along a given batch dimension.
        """
        if dim < 0:
            dim += len(parts[0].batch_shape)
        args = [torch.cat([getattr(g, attr) for g in parts], dim=dim)
                for attr in ["log_normalizer", "info_vec", "precision"]]
        return Gaussian(*args)

    def event_pad(self, left=0, right=0):
        """
        Pad along event dimension.
        """
        lr = (left, right)
        log_normalizer = self.log_normalizer
        info_vec = pad(self.info_vec, lr)
        precision = pad(self.precision, lr + lr)
        return Gaussian(log_normalizer, info_vec, precision)

    def event_permute(self, perm):
        """
        Permute along event dimension.
        """
        assert isinstance(perm, torch.Tensor)
        assert perm.shape == (self.dim(),)
        info_vec = self.info_vec[..., perm]
        precision = self.precision[..., perm][..., perm, :]
        return Gaussian(self.log_normalizer, info_vec, precision)

    def __add__(self, other):
        """
        Adds two Gaussians in log-density space.
        """
        assert isinstance(other, Gaussian)
        assert self.dim() == other.dim()
        return Gaussian(self.log_normalizer + other.log_normalizer,
                        self.info_vec + other.info_vec,
                        self.precision + other.precision)

    def log_density(self, value):
        """
        Evaluate the log density of this Gaussian at a point value::

            -0.5 * value.T @ precision @ value + value.T @ info_vec + log_normalizer

        This is mainly used for testing.
        """
        if value.size(-1) == 0:
            batch_shape = broadcast_shape(value.shape[:-1], self.batch_shape)
            return self.log_normalizer.expand(batch_shape)
        result = (-0.5) * self.precision.matmul(value.unsqueeze(-1)).squeeze(-1)
        result = result + self.info_vec
        result = (value * result).sum(-1)
        return result + self.log_normalizer

    def condition(self, value):
        """
        Condition this Gaussian on a trailing subset of its state.
        This should satisfy::

            g.condition(y).dim() == g.dim() - y.size(-1)

        Note that since this is a non-normalized Gaussian, we include the
        density of ``y`` in the result. Thus :meth:`condition` is similar to a
        ``functools.partial`` binding of arguments::

            left = x[..., :n]
            right = x[..., n:]
            g.log_density(x) == g.condition(right).log_density(left)
        """
        assert isinstance(value, torch.Tensor)
        assert value.size(-1) <= self.info_vec.size(-1)

        n = self.dim() - value.size(-1)
        info_a = self.info_vec[..., :n]
        info_b = self.info_vec[..., n:]
        P_aa = self.precision[..., :n, :n]
        P_ab = self.precision[..., :n, n:]
        P_bb = self.precision[..., n:, n:]
        b = value

        info_vec = info_a - P_ab.matmul(b.unsqueeze(-1)).squeeze(-1)
        precision = P_aa
        log_normalizer = (self.log_normalizer +
                          -0.5 * P_bb.matmul(b.unsqueeze(-1)).squeeze(-1).mul(b).sum(-1) +
                          b.mul(info_b).sum(-1))
        return Gaussian(log_normalizer, info_vec, precision)

    def marginalize(self, left=0, right=0):
        """
        Marginalizing out variables on either side of the event dimension::

            g.marginalize(left=n).event_logsumexp() = g.logsumexp()
            g.marginalize(right=n).event_logsumexp() = g.logsumexp()

        and for data ``x``:

            g.condition(x).event_logsumexp()
              = g.marginalize(left=g.dim() - x.size(-1)).log_density(x)
        """
        if left == 0 and right == 0:
            return self
        if left > 0 and right > 0:
            raise NotImplementedError
        n = self.dim()
        n_b = left + right
        a = slice(left, n - right)  # preserved
        b = slice(None, left) if left else slice(n - right, None)

        P_aa = self.precision[..., a, a]
        P_ba = self.precision[..., b, a]
        P_bb = self.precision[..., b, b]
        P_b = P_bb.cholesky()
        P_a = P_ba.triangular_solve(P_b, upper=False).solution
        P_at = P_a.transpose(-1, -2)
        precision = P_aa - P_at.matmul(P_a)

        info_a = self.info_vec[..., a]
        info_b = self.info_vec[..., b]
        b_tmp = info_b.unsqueeze(-1).triangular_solve(P_b, upper=False).solution
        info_vec = info_a - P_at.matmul(b_tmp).squeeze(-1)

        log_normalizer = (self.log_normalizer +
                          0.5 * n_b * math.log(2 * math.pi) -
                          P_b.diagonal(dim1=-2, dim2=-1).log().sum(-1) +
                          0.5 * b_tmp.squeeze(-1).pow(2).sum(-1))
        return Gaussian(log_normalizer, info_vec, precision)

    def event_logsumexp(self):
        """
        Integrates out all latent state (i.e. operating on event dimensions).
        """
        n = self.dim()
        chol_P = self.precision.cholesky()
        chol_P_u = self.info_vec.unsqueeze(-1).triangular_solve(chol_P, upper=False).solution.squeeze(-1)
        u_P_u = chol_P_u.pow(2).sum(-1)
        return (self.log_normalizer + 0.5 * n * math.log(2 * math.pi) + 0.5 * u_P_u -
                chol_P.diagonal(dim1=-2, dim2=-1).log().sum(-1))


class AffineNormal:
    """
    Represents a conditional diagonal normal distribution over a random
    variable ``Y`` whose mean is an affine function of a random variable ``X``.
    The likelihood of ``X`` is thus::

        AffineNormal(matrix, loc, scale).condition(y).log_density(x)

    which is equivalent to::

        Normal(x @ matrix + loc, scale).to_event(1).log_prob(y)

    :param torch.Tensor matrix: A transformation from ``X`` to ``Y``.
        Should have rightmost shape ``(x_dim, y_dim)``.
    :param torch.Tensor loc: A constant offset for ``Y``'s mean.
        Should have rightmost shape ``(y_dim,)``.
    :param torch.Tensor scale: Standard deviation for ``Y``.
        Should have rightmost shape ``(y_dim,)``.
    """
    def __init__(self, matrix, loc, scale):
        assert loc.shape == scale.shape
        x_dim, y_dim = matrix.shape[-2:]
        self.matrix = matrix
        self.loc = loc
        self.scale = scale

    def condition(self, value):
        """
        Condition on a ``Y`` value.

        :param torch.Tensor value: A value of ``Y``.
        :return Gaussian: A gaussian likelihood over ``X``.
        """
        assert value.size(-1) == self.loc.size(-1)
        prec_sqrt = self.matrix / self.scale.unsqueeze(-2)
        precision = prec_sqrt.matmul(prec_sqrt.transpose(-1, -2))
        delta = (value - self.loc) / self.scale
        info_vec = prec_sqrt.matmul(delta.unsqueeze(-1)).squeeze(-1)
        log_normalizer = (-0.5 * self.loc.size(-1) * math.log(2 * math.pi)
                          - 0.5 * delta.pow(2).sum(-1) - self.scale.log().sum(-1))
        return Gaussian(log_normalizer, info_vec, precision)

    def to_gaussian(self):
        mvn = torch.distributions.MultivariateNormal(self.loc, scale_tril=self.scale.diag_embed())
        return matrix_and_mvn_to_gaussian(self.matrix, mvn)

    def __add__(self, other):
        return self.to_gaussian() + other


def mvn_to_gaussian(mvn):
    """
    Convert a MultivariateNormal distribution to a Gaussian.

    :param ~torch.distributions.MultivariateNormal mvn: A multivariate normal distribution.
    :return: An equivalent Gaussian object.
    :rtype: ~pyro.ops.gaussian.Gaussian
    """
    assert (isinstance(mvn, torch.distributions.MultivariateNormal) or
            (isinstance(mvn, torch.distributions.Independent) and
             isinstance(mvn.base_dist, torch.distributions.Normal)))
    if isinstance(mvn, torch.distributions.Independent):
        mvn = mvn.base_dist
        precision_diag = mvn.scale.pow(-2)
        precision = precision_diag.diag_embed()
        info_vec = mvn.loc * precision_diag
        scale_diag = mvn.scale
    else:
        precision = mvn.precision_matrix
        info_vec = precision.matmul(mvn.loc.unsqueeze(-1)).squeeze(-1)
        scale_diag = mvn.scale_tril.diagonal(dim1=-2, dim2=-1)

    n = mvn.loc.size(-1)
    log_normalizer = (-0.5 * n * math.log(2 * math.pi) +
                      -0.5 * (info_vec * mvn.loc).sum(-1) -
                      scale_diag.log().sum(-1))
    return Gaussian(log_normalizer, info_vec, precision)


def matrix_and_mvn_to_gaussian(matrix, mvn):
    """
    Convert a noisy affine function to a Gaussian. The noisy affine function is defined as::

        y = x @ matrix + mvn.sample()

    :param ~torch.Tensor matrix: A matrix with rightmost shape ``(x_dim, y_dim)``.
    :param ~torch.distributions.MultivariateNormal mvn: A multivariate normal distribution.
    :return: A Gaussian with broadcasted batch shape and ``.dim() == x_dim + y_dim``.
    :rtype: ~pyro.ops.gaussian.Gaussian
    """
    assert (isinstance(mvn, torch.distributions.MultivariateNormal) or
            (isinstance(mvn, torch.distributions.Independent) and
             isinstance(mvn.base_dist, torch.distributions.Normal)))
    assert isinstance(matrix, torch.Tensor)
    x_dim, y_dim = matrix.shape[-2:]
    assert mvn.event_shape == (y_dim,)
    batch_shape = broadcast_shape(matrix.shape[:-2], mvn.batch_shape)
    matrix = matrix.expand(batch_shape + (x_dim, y_dim))
    mvn = mvn.expand(batch_shape)

    # Handle diagonal normal distributions as an efficient special case.
    if isinstance(mvn, torch.distributions.Independent):
        return AffineNormal(matrix, mvn.base_dist.loc, mvn.base_dist.scale)

    y_gaussian = mvn_to_gaussian(mvn)
    P_yy = y_gaussian.precision
    neg_P_xy = matrix.matmul(P_yy)
    P_xy = -neg_P_xy
    P_yx = P_xy.transpose(-1, -2)
    P_xx = neg_P_xy.matmul(matrix.transpose(-1, -2))
    precision = torch.cat([torch.cat([P_xx, P_xy], -1),
                           torch.cat([P_yx, P_yy], -1)], -2)
    info_y = y_gaussian.info_vec
    info_x = -matrix.matmul(info_y.unsqueeze(-1)).squeeze(-1)
    info_vec = torch.cat([info_x, info_y], -1)
    log_normalizer = y_gaussian.log_normalizer

    result = Gaussian(log_normalizer, info_vec, precision)
    assert result.batch_shape == batch_shape
    assert result.dim() == x_dim + y_dim
    return result


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
