# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.distributions.utils import lazy_property
from torch.nn.functional import pad

from pyro.distributions.util import broadcast_shape
from pyro.ops.tensor_utils import cholesky, matmul, matvecmul, triangular_solve


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
        if isinstance(other, Gaussian):
            assert self.dim() == other.dim()
            return Gaussian(self.log_normalizer + other.log_normalizer,
                            self.info_vec + other.info_vec,
                            self.precision + other.precision)
        if isinstance(other, (int, float, torch.Tensor)):
            return Gaussian(self.log_normalizer + other, self.info_vec, self.precision)
        raise ValueError("Unsupported type: {}".format(type(other)))

    def __sub__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            return Gaussian(self.log_normalizer - other, self.info_vec, self.precision)
        raise ValueError("Unsupported type: {}".format(type(other)))

    def log_density(self, value):
        """
        Evaluate the log density of this Gaussian at a point value::

            -0.5 * value.T @ precision @ value + value.T @ info_vec + log_normalizer

        This is mainly used for testing.
        """
        if value.size(-1) == 0:
            batch_shape = broadcast_shape(value.shape[:-1], self.batch_shape)
            return self.log_normalizer.expand(batch_shape)
        result = (-0.5) * matvecmul(self.precision, value)
        result = result + self.info_vec
        result = (value * result).sum(-1)
        return result + self.log_normalizer

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sampler.
        """
        P_chol = cholesky(self.precision)
        loc = self.info_vec.unsqueeze(-1).cholesky_solve(P_chol).squeeze(-1)
        shape = sample_shape + self.batch_shape + (self.dim(), 1)
        noise = torch.randn(shape, dtype=loc.dtype, device=loc.device)
        noise = triangular_solve(noise, P_chol, upper=False, transpose=True).squeeze(-1)
        return loc + noise

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
        right = value.size(-1)
        dim = self.dim()
        assert right <= dim

        n = dim - right
        info_a = self.info_vec[..., :n]
        info_b = self.info_vec[..., n:]
        P_aa = self.precision[..., :n, :n]
        P_ab = self.precision[..., :n, n:]
        P_bb = self.precision[..., n:, n:]
        b = value

        info_vec = info_a - matvecmul(P_ab, b)
        precision = P_aa
        log_normalizer = (self.log_normalizer +
                          -0.5 * matvecmul(P_bb, b).mul(b).sum(-1) +
                          b.mul(info_b).sum(-1))
        return Gaussian(log_normalizer, info_vec, precision)

    def left_condition(self, value):
        """
        Condition this Gaussian on a leading subset of its state.
        This should satisfy::

            g.condition(y).dim() == g.dim() - y.size(-1)

        Note that since this is a non-normalized Gaussian, we include the
        density of ``y`` in the result. Thus :meth:`condition` is similar to a
        ``functools.partial`` binding of arguments::

            left = x[..., :n]
            right = x[..., n:]
            g.log_density(x) == g.left_condition(left).log_density(right)
        """
        assert isinstance(value, torch.Tensor)
        left = value.size(-1)
        dim = self.dim()
        assert left <= dim

        perm = torch.cat([torch.arange(left, dim, device=value.device),
                          torch.arange(left, device=value.device)])
        return self.event_permute(perm).condition(value)

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
        P_b = cholesky(P_bb)
        P_a = triangular_solve(P_ba, P_b, upper=False)
        P_at = P_a.transpose(-1, -2)
        precision = P_aa - matmul(P_at, P_a)

        info_a = self.info_vec[..., a]
        info_b = self.info_vec[..., b]
        b_tmp = triangular_solve(info_b.unsqueeze(-1), P_b, upper=False)
        info_vec = info_a - matmul(P_at, b_tmp).squeeze(-1)

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
        chol_P = cholesky(self.precision)
        chol_P_u = triangular_solve(self.info_vec.unsqueeze(-1), chol_P, upper=False).squeeze(-1)
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
        assert matrix.shape[:-2] == loc.shape[:-1]
        assert matrix.size(-1) == loc.size(-1)
        self.matrix = matrix
        self.loc = loc
        self.scale = scale
        self._gaussian = None

    @lazy_property
    def batch_shape(self):
        return self.matrix.shape[:-2]

    def condition(self, value):
        if value.size(-1) == self.loc.size(-1):
            prec_sqrt = self.matrix / self.scale.unsqueeze(-2)
            precision = matmul(prec_sqrt, prec_sqrt.transpose(-1, -2))
            delta = (value - self.loc) / self.scale
            info_vec = matvecmul(prec_sqrt, delta)
            log_normalizer = (-0.5 * self.loc.size(-1) * math.log(2 * math.pi)
                              - 0.5 * delta.pow(2).sum(-1) - self.scale.log().sum(-1))
            return Gaussian(log_normalizer, info_vec, precision)
        else:
            return self.to_gaussian().condition(value)

    def left_condition(self, value):
        """
        If ``value.size(-1) == x_dim``, this returns a Normal distribution with
        ``event_dim=1``. After applying this method, the cost to draw a sample is
        ``O(y_dim)`` instead of ``O(y_dim ** 3)``.
        """
        if value.size(-1) == self.matrix.size(-2):
            loc = matvecmul(self.matrix.transpose(-1, -2), value) + self.loc
            matrix = value.new_zeros(loc.shape[:-1] + (0, loc.size(-1)))
            scale = self.scale.expand(loc.shape)
            return AffineNormal(matrix, loc, scale)
        else:
            return self.to_gaussian().left_condition(value)

    def rsample(self, sample_shape=torch.Size()):
        """
        Reparameterized sampler.
        """
        if self.matrix.size(-2) > 0:
            raise NotImplementedError
        shape = sample_shape + self.batch_shape + self.loc.shape[-1:]
        noise = torch.randn(shape, dtype=self.loc.dtype, device=self.loc.device)
        return self.loc + noise * self.scale

    def to_gaussian(self):
        if self._gaussian is None:
            mvn = torch.distributions.Independent(
                torch.distributions.Normal(self.loc, scale=self.scale), 1)
            y_gaussian = mvn_to_gaussian(mvn)
            self._gaussian = _matrix_and_gaussian_to_gaussian(self.matrix, y_gaussian)
        return self._gaussian

    def expand(self, batch_shape):
        matrix = self.matrix.expand(batch_shape + self.matrix.shape[-2:])
        loc = self.loc.expand(batch_shape + self.loc.shape[-1:])
        scale = self.scale.expand(batch_shape + self.scale.shape[-1:])
        return AffineNormal(matrix, loc, scale)

    def reshape(self, batch_shape):
        matrix = self.matrix.reshape(batch_shape + self.matrix.shape[-2:])
        loc = self.loc.reshape(batch_shape + self.loc.shape[-1:])
        scale = self.scale.reshape(batch_shape + self.scale.shape[-1:])
        return AffineNormal(matrix, loc, scale)

    def __getitem__(self, index):
        assert isinstance(index, tuple)
        matrix = self.matrix[index + (slice(None), slice(None))]
        loc = self.loc[index + (slice(None),)]
        scale = self.scale[index + (slice(None),)]
        return AffineNormal(matrix, loc, scale)

    def event_permute(self, perm):
        return self.to_gaussian().event_permute(perm)

    def __add__(self, other):
        return self.to_gaussian() + other

    def marginalize(self, left=0, right=0):
        if left == 0 and right == self.loc.size(-1):
            n = self.matrix.size(-2)
            precision = self.scale.new_zeros(self.batch_shape + (n, n))
            info_vec = self.scale.new_zeros(self.batch_shape + (n,))
            log_normalizer = self.scale.new_zeros(self.batch_shape)
            return Gaussian(log_normalizer, info_vec, precision)
        else:
            return self.to_gaussian().marginalize(left, right)


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
        info_vec = matvecmul(precision, mvn.loc)
        scale_diag = mvn.scale_tril.diagonal(dim1=-2, dim2=-1)

    n = mvn.loc.size(-1)
    log_normalizer = (-0.5 * n * math.log(2 * math.pi) +
                      -0.5 * (info_vec * mvn.loc).sum(-1) -
                      scale_diag.log().sum(-1))
    return Gaussian(log_normalizer, info_vec, precision)


def _matrix_and_gaussian_to_gaussian(matrix, y_gaussian):
    P_yy = y_gaussian.precision
    neg_P_xy = matmul(matrix, P_yy)
    P_xy = -neg_P_xy
    P_yx = P_xy.transpose(-1, -2)
    P_xx = matmul(neg_P_xy, matrix.transpose(-1, -2))
    precision = torch.cat([torch.cat([P_xx, P_xy], -1),
                           torch.cat([P_yx, P_yy], -1)], -2)
    info_y = y_gaussian.info_vec
    info_x = -matvecmul(matrix, info_y)
    info_vec = torch.cat([info_x, info_y], -1)
    log_normalizer = y_gaussian.log_normalizer

    result = Gaussian(log_normalizer, info_vec, precision)
    return result


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
    result = _matrix_and_gaussian_to_gaussian(matrix, y_gaussian)
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
        L = cholesky(Pbb + Qbb)
        LinvB = triangular_solve(B, L, upper=False)
        LinvBt = LinvB.transpose(-2, -1)
        Linvb = triangular_solve(b.unsqueeze(-1), L, upper=False)

        precision = precision - matmul(LinvBt, LinvB)
        # NB: precision might not be invertible for getting mean = precision^-1 @ info_vec
        if na + nc > 0:
            info_vec = info_vec - matmul(LinvBt, Linvb).squeeze(-1)
        logdet = torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)
        diff = 0.5 * nb * math.log(2 * math.pi) + 0.5 * Linvb.squeeze(-1).pow(2).sum(-1) - logdet
        log_normalizer = log_normalizer + diff

    return Gaussian(log_normalizer, info_vec, precision)
