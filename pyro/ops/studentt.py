import math

import torch
from torch.distributions.utils import lazy_property
from torch.nn.functional import pad

from pyro.distributions.multivariate_studentt import MultivariateStudentT
from pyro.distributions.util import broadcast_shape


def _precision_to_scale_tril(P):
    Lf = torch.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    L = torch.triangular_solve(torch.eye(P.shape[-1], dtype=P.dtype, device=P.device),
                               L_inv, upper=False)[0]
    return L


class Gamma:
    """
    Non-normalized Gamma distribution.
    """
    def __init__(self, log_normalizer, alpha, beta):
        self.log_normalizer = log_normalizer
        self.alpha = alpha
        self.beta = beta

    def log_density(self, s):
        """
        Non-normalized log probability of Gamma distribution.

        This is mainly used for testing.
        """
        return self.log_normalizer + (self.alpha - 1) * s.log() - self.beta * s

    def logsumexp(self):
        """
        Integrates out the latent variable.
        """
        return self.log_normalizer + torch.lgamma(self.alpha) - self.alpha * self.beta.log()


class GaussianGamma:
    """
    Non-normalized GaussianGamma distribution:

        GaussianGamma(x, s) ~ (alpha + 0.5 * dim - 1) * log(s)
                              - beta * s - s * 0.5 * info_vec.T @ inv(precision) @ info_vec)
                              - s * 0.5 * x.T @ precision @ x + s * x.T @ info_vec,

    which will be reparameterized as

        GaussianGamma(x, s) =: alpha' * log(s) + s * (-0.5 * x.T @ precision @ x + x.T @ info_vec - beta').

    This represents an arbitrary semidefinite quadratic function, which can be
    interpreted as a rank-deficient scaled Gaussian distribution. The precision
    matrix may have zero eigenvalues, thus it may be impossible to work
    directly with the covariance matrix. The `s` variable plays the role of a multiplier such that

        p(x | s) ~ Gaussian(s * info_vec, s * precision).

    :param torch.Tensor log_normalizer: a normalization constant, which is mainly used to keep
        track of normalization terms during contractions.
    :param torch.Tensor info_vec: information vector, which is a scaled version of the mean
        ``info_vec = precision @ mean``. We use this represention to make gaussian contraction
        fast and stable.
    :param torch.Tensor precision: precision matrix of this gaussian.
    :param torch.Tensor alpha: reparameterized shape parameter of the marginal Gamma distribution of
        `s`. The shape parameter Gamma.alpha is reparameterized by:

            alpha = Gamma.alpha + 0.5 * dim - 1

    :param torch.Tensor beta: reparameterized rate parameter of the marginal Gamma distribution of
        `s`. The rate parameter Gamma.beta is reparameterized by:

            beta = Gamma.beta + 0.5 * info_vec.T @ inv(precision) @ info_vec
    """
    def __init__(self, log_normalizer, info_vec, precision, alpha, beta):
        # NB: using info_vec instead of mean to deal with rank-deficient problem
        assert info_vec.dim() >= 1
        assert precision.dim() >= 2
        assert precision.shape[-2:] == info_vec.shape[-1:] * 2
        self.log_normalizer = log_normalizer
        self.info_vec = info_vec
        self.precision = precision
        self.alpha = alpha
        self.beta = beta

    def dim(self):
        return self.info_vec.size(-1)

    @lazy_property
    def batch_shape(self):
        return broadcast_shape(self.log_normalizer.shape,
                               self.info_vec.shape[:-1],
                               self.precision.shape[:-2],
                               self.alpha.shape,
                               self.beta.shape)

    def expand(self, batch_shape):
        n = self.dim()
        log_normalizer = self.log_normalizer.expand(batch_shape)
        info_vec = self.info_vec.expand(batch_shape + (n,))
        precision = self.precision.expand(batch_shape + (n, n))
        alpha = self.alpha.expand(batch_shape)
        beta = self.beta.expand(batch_shape)
        return GaussianGamma(log_normalizer, info_vec, precision, alpha, beta)

    def reshape(self, batch_shape):
        n = self.dim()
        log_normalizer = self.log_normalizer.reshape(batch_shape)
        info_vec = self.info_vec.reshape(batch_shape + (n,))
        precision = self.precision.reshape(batch_shape + (n, n))
        alpha = self.alpha.reshape(batch_shape)
        beta = self.beta.reshape(batch_shape)
        return GaussianGamma(log_normalizer, info_vec, precision, alpha, beta)

    def __getitem__(self, index):
        """
        Index into the batch_shape of a GaussianGamma.
        """
        assert isinstance(index, tuple)
        log_normalizer = self.log_normalizer[index]
        info_vec = self.info_vec[index + (slice(None),)]
        precision = self.precision[index + (slice(None), slice(None))]
        alpha = self.alpha[index]
        beta = self.beta[index]
        return GaussianGamma(log_normalizer, info_vec, precision, alpha, beta)

    @staticmethod
    def cat(parts, dim=0):
        """
        Concatenate a list of GaussianGammas along a given batch dimension.
        """
        if dim < 0:
            dim += len(parts[0].batch_shape)
        args = [torch.cat([getattr(g, attr) for g in parts], dim=dim)
                for attr in ["log_normalizer", "info_vec", "precision", "alpha", "beta"]]
        return GaussianGamma(*args)

    def event_pad(self, left=0, right=0):
        """
        Pad along event dimension.
        """
        lr = (left, right)
        info_vec = pad(self.info_vec, lr)
        precision = pad(self.precision, lr + lr)
        # no change for alpha, beta because we are working with reparameterized version
        return GaussianGamma(self.log_normalizer, info_vec, precision, self.alpha, self.beta)

    def event_permute(self, perm):
        """
        Permute along event dimension.
        """
        assert isinstance(perm, torch.Tensor)
        assert perm.shape == (self.dim(),)
        info_vec = self.info_vec[..., perm]
        precision = self.precision[..., perm][..., perm, :]
        return GaussianGamma(self.log_normalizer, info_vec, precision, self.alpha, self.beta)

    def __add__(self, other):
        """
        Adds two GaussianGammas in log-density space.
        """
        assert isinstance(other, GaussianGamma)
        assert self.dim() == other.dim()
        return GaussianGamma(self.log_normalizer + other.log_normalizer,
                             self.info_vec + other.info_vec,
                             self.precision + other.precision,
                             self.alpha + other.alpha,
                             self.beta + other.beta)

    def log_density(self, value, s):
        """
        Evaluate the log density of this GaussianGamma at a point value::

            alpha * log(s) + s * (-0.5 * value.T @ precision @ value + value.T @ info_vec - beta) + log_normalizer

        This is mainly used for testing.
        """
        if value.size(-1) == 0:
            batch_shape = broadcast_shape(value.shape[:-1], s.shape, self.batch_shape)
            return self.alpha * s.log() - self.beta * s + self.log_normalizer.expand(batch_shape)
        result = (-0.5) * self.precision.matmul(value.unsqueeze(-1)).squeeze(-1)
        result = result + self.info_vec
        result = (value * result).sum(-1)
        return self.alpha * s.log() + (result - self.beta) * s + self.log_normalizer

    def condition(self, value):
        """
        Condition the Gaussian component on a trailing subset of its state.
        This should satisfy::

            g.condition(y).dim() == g.dim() - y.size(-1)

        Note that since this is a non-normalized Gaussian, we include the
        density of ``y`` in the result. Thus :meth:`condition` is similar to a
        ``functools.partial`` binding of arguments::

            left = x[..., :n]
            right = x[..., n:]
            g.log_density(x, s) == g.condition(right).log_density(left, s)
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

        log_normalizer = self.log_normalizer
        alpha = self.alpha
        beta = self.beta + 0.5 * P_bb.matmul(b.unsqueeze(-1)).squeeze(-1).mul(b).sum(-1) - b.mul(info_b).sum(-1)
        return GaussianGamma(log_normalizer, info_vec, precision, alpha, beta)

    def marginalize(self, left=0, right=0):
        """
        Marginalizing out variables on either side of the event dimension::

            g.marginalize(left=n).event_logsumexp() = g.event_logsumexp()
            g.marginalize(right=n).event_logsumexp() = g.event_logsumexp()

        and for data ``x``:

            g.condition(x).event_logsumexp().log_density(s)
              = g.marginalize(left=g.dim() - x.size(-1)).log_density(x, s)
        """
        # NB: the easiest way to think about this process is to consider GaussianGamma
        # as a Gaussian with precision and info_vec scaled by `s`.
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
        info_vec = info_a
        if n_b < n:
            info_vec = info_vec - P_at.matmul(b_tmp).squeeze(-1)

        alpha = self.alpha - 0.5 * n_b
        beta = self.beta - 0.5 * b_tmp.squeeze(-1).pow(2).sum(-1)
        log_normalizer = (self.log_normalizer +
                          0.5 * n_b * math.log(2 * math.pi) -
                          P_b.diagonal(dim1=-2, dim2=-1).log().sum(-1))
        return GaussianGamma(log_normalizer, info_vec, precision, alpha, beta)

    def compound(self):
        """
        Integrates out the latent multiplier `s`. The result will be a
        Student-T distribution.
        """
        alpha = self.alpha - 0.5 * self.dim() + 1
        scale_tril = _precision_to_scale_tril(self.precision)
        scale_tril_t_u = scale_tril.transpose(-1, -2).matmul(self.info_vec.unsqueeze(-1)).squeeze(-1)
        u_Pinv_u = scale_tril_t_u.pow(2).sum(-1)
        beta = self.beta - 0.5 * u_Pinv_u

        loc = scale_tril.matmul(scale_tril_t_u.unsqueeze(-1)).squeeze(-1)
        scale_tril = scale_tril * (beta / alpha).sqrt().unsqueeze(-1).unsqueeze(-1)
        return MultivariateStudentT(2 * alpha, loc, scale_tril)

    def event_logsumexp(self):
        """
        Integrates out all latent state (i.e. operating on event dimensions) of Gaussian component.
        """
        n = self.dim()
        chol_P = self.precision.cholesky()
        chol_P_u = self.info_vec.unsqueeze(-1).triangular_solve(chol_P, upper=False).solution.squeeze(-1)
        u_P_u = chol_P_u.pow(2).sum(-1)
        # considering GaussianGamma as a Gaussian with precision = s * precision, info_vec = s * info_vec,
        # marginalize x variable, we get
        #   logsumexp(s) = alpha' * log(s) - s * beta' + 0.5 n * log(2 pi) + 0.5 s * uPu - 0.5 * log|P| - 0.5 n * log(s)
        # use the original parameterization of Gamma, we get
        #   logsumexp(s) = (alpha - 1) * log(s) - s * beta + 0.5 n * log(2 pi) - 0.5 * |P|
        # Note that `(alpha - 1) * log(s) - s * beta` is unnormalized log_prob of Gamma(alpha, beta)
        alpha = self.alpha - 0.5 * n + 1
        beta = self.beta - 0.5 * u_P_u
        log_normalizer_tmp = 0.5 * n * math.log(2 * math.pi) - chol_P.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return Gamma(self.log_normalizer + log_normalizer_tmp, alpha, beta)


def mvt_to_gaussian_gamma(mvt, return_conditional=False):
    """
    Convert a MultivariateStudentT (MVT) distribution to a GaussianGamma.

        p(x) ~ MVT(df, info_vec, precision)
        p(x | s) ~ Gaussian(s * info_vec, s * precision)
        p(s) ~ Gamma(df / 2, df / 2)
        p(x, s) ~ GaussianGamma(info_vec, precison, df / 2, df / 2)

    :param ~pyro.distributions.MultivariateStudentT mvt: A multivariate student-t distribution.
    :param bool return_conditional: whether to return the conditional or the joint distribution.
        If True, we return p(x | s). Otherwise, we return p(x, s).
    :return: A GaussianGamma object which is equivalent to the MVT when marginalized out the
        multiplier `s`.
    :rtype: ~pyro.ops.studentt.GaussianGamma
    """
    n = mvt.loc.size(-1)
    precision = mvt.precision_matrix
    info_vec = precision.matmul(mvt.loc.unsqueeze(-1)).squeeze(-1)

    # First, we model p(x | s).
    # Here, we use reparameterized versions of alpha = 1, beta = 0
    # Note that Gamma(1, 0).log_density is a constant function. In other works, with these values of
    # alpha, beta, we say that there is no prior for `s`. This is similar to Gaussian with zero
    # info_vec and zero precision.
    alpha = torch.full(mvt.batch_shape, n / 2.0,
                       device=info_vec.device, dtype=info_vec.dtype)
    beta = 0.5 * (info_vec * mvt.loc).sum(-1)
    log_normalizer = -0.5 * n * math.log(2 * math.pi) - mvt.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)

    if not return_conditional:
        # reparameterized version of alpha = beta = 0.5 dof
        half_df = 0.5 * mvt.df
        alpha = alpha - 1 + half_df
        beta = beta + half_df
        log_normalizer = log_normalizer - Gamma(0., half_df, half_df).logsumexp()
    return GaussianGamma(log_normalizer, info_vec, precision, alpha, beta)


def matrix_and_mvt_to_gaussian_gamma(matrix, mvt, return_conditional=False):
    """
    Convert a noisy affine function to a GaussianGamma. The noisy affine function is defined as::

        y = x @ matrix + mvt.sample()

    :param ~torch.Tensor matrix: A matrix with rightmost shape ``(x_dim, y_dim)``.
    :param ~pyro.distributions.MultivariateStudentT mvt: A multivariate student-t distribution.
    :param bool return_conditional: whether to return the conditional or the joint distribution.
        If True, we return p(x | s). Otherwise, we return p(x, s).
    :return: A GaussianGamma with broadcasted batch shape and ``.dim() == x_dim + y_dim``.
    :rtype: ~pyro.ops.studentt.GaussianGamma
    """
    assert isinstance(mvt, MultivariateStudentT)
    assert isinstance(matrix, torch.Tensor)
    x_dim, y_dim = matrix.shape[-2:]
    assert mvt.event_shape == (y_dim,)
    batch_shape = broadcast_shape(matrix.shape[:-2], mvt.batch_shape)
    matrix = matrix.expand(batch_shape + (x_dim, y_dim))
    mvt = mvt.expand(batch_shape)

    y_gaussian_gamma = mvt_to_gaussian_gamma(mvt, return_conditional=return_conditional)
    P_yy = y_gaussian_gamma.precision
    neg_P_xy = matrix.matmul(P_yy)
    P_xy = -neg_P_xy
    P_yx = P_xy.transpose(-1, -2)
    P_xx = neg_P_xy.matmul(matrix.transpose(-1, -2))
    precision = torch.cat([torch.cat([P_xx, P_xy], -1),
                           torch.cat([P_yx, P_yy], -1)], -2)
    info_y = y_gaussian_gamma.info_vec
    info_x = -matrix.matmul(info_y.unsqueeze(-1)).squeeze(-1)
    info_vec = torch.cat([info_x, info_y], -1)
    log_normalizer = y_gaussian_gamma.log_normalizer
    alpha = y_gaussian_gamma.alpha
    beta = y_gaussian_gamma.beta

    result = GaussianGamma(log_normalizer, info_vec, precision, alpha, beta)
    assert result.batch_shape == batch_shape
    assert result.dim() == x_dim + y_dim
    return result


def gaussian_gamma_tensordot(x, y, dims=0):
    """
    Computes the integral over two GaussianGammas:

        `(x @ y)((a,c),s) = log(integral(exp(x((a,b),s) + y((b,c),s)), b))`,

    where `x` is a gaussian over variables (a,b), `y` is a gaussian over variables
    (b,c), (a,b,c) can each be sets of zero or more variables, and `dims` is the size of b.

    :param x: a GaussianGamma instance
    :param y: a GaussianGamma instance
    :param dims: number of variables to contract
    """
    assert isinstance(x, GaussianGamma)
    assert isinstance(y, GaussianGamma)
    na = x.dim() - dims
    nb = dims
    nc = y.dim() - dims
    assert na >= 0
    assert nb >= 0
    assert nc >= 0

    device = x.info_vec.device
    perm = torch.cat([
        torch.arange(na, device=device),
        torch.arange(x.dim(), x.dim() + nc, device=device),
        torch.arange(na, x.dim(), device=device)])
    return (x.event_pad(right=nc) + y.event_pad(left=na)).event_permute(perm).marginalize(right=nb)
