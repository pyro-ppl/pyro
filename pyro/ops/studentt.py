import torch
from torch.distributions.utils import lazy_property

import pyro
from pyro.distributions.util import broadcast_shape
from pyro.ops.gamma_gaussian import Gamma, GammaGaussian, gamma_and_mvn_to_gamma_gaussian


def _absolute_central_moment_matching(st, new_df):
    """
    Approximates a StudentT by another one with different degree of freedom.
    """
    # Ref: Moments of Student's t-distribution: A Unified Approach,
    # J. Lars Kirkby, Dang Nguyen, Duy Nguyen https://arxiv.org/abs/1912.01607
    # We want to find s such that
    #   St(df, m, P) ~ St(new_df, m', P')
    # Matching the mean gives us m' = m.
    # We need to find s such that P' := sP
    # i.e. St(df, 0, I) ~ St(new_df, 0, sI)
    # Given a moment with order k1, k2,... such that sum(k_i) = 1,
    # we will use the absolute moment formula in Theorem 3 of the above reference
    # and match the corresponding moments of two distributions.
    half_df = 0.5 * st.df
    half_new_df = 0.5 * new_df
    s = new_df / st.df * torch.exp(2 * (
        torch.lgamma(half_new_df - 0.5) - torch.lgamma(half_new_df)
        - torch.lgamma(half_df - 0.5) + torch.lgamma(half_df)))
    # adjust log_normalizer
    log_normalizer = st.log_normalizer + Gamma(0., half_df, half_df).logsumexp() - \
        Gamma(0., half_new_df, half_new_df).logsumexp() - 0.5 * st.rank * s.log()
    return StudentT(log_normalizer,
                    st.info_vec * s.unsqueeze(-1),
                    st.precision * s.unsqueeze(-1).unsqueeze(-1),
                    new_df.expand(st.batch_shape),
                    st.rank)


def _precision_to_scale_tril(P):
    Lf = torch.cholesky(torch.flip(P, (-2, -1)))
    L_inv = torch.transpose(torch.flip(Lf, (-2, -1)), -2, -1)
    L = torch.triangular_solve(torch.eye(P.shape[-1], dtype=P.dtype, device=P.device),
                               L_inv, upper=False)[0]
    return L


class StudentT(GammaGaussian):
    """
    Non-normalized StudentT distribution:

        StudentT(x) ~ Integral(p(x, s), s),

    where

        p(x, s) = GammaGaussian(log_normalizer, info_vec, precision, alpha, beta).

    Note that different StudentT instances can have the same log density.

    The `s` variable plays the role of a mixing variable such that

        p(x | s) ~ Gaussian(s * info_vec, s * precision).

    Conditioned on `s`, this represents an arbitrary semidefinite quadratic function,
    which can be interpreted as a rank-deficient StudentT distribution.
    The precision matrix may have zero eigenvalues, thus it may be impossible
    to work directly with the covariance matrix.

    :param torch.Tensor log_normalizer: a normalization constant, which is mainly used to keep
        track of normalization terms during contractions.
    :param torch.Tensor info_vec: information vector, which is a scaled version of the mean
        ``info_vec = precision @ mean``. We use this represention to make student-t contraction
        fast and stable.
    :param torch.Tensor precision: precision matrix of the joint GammaGaussian.
    :param torch.Tensor alpha: alpha parameter of the joint GammaGaussian.
    :param torch.Tensor beta: beta parameter of the joint GammaGaussian.
    """
    def __init__(self, log_normalizer, info_vec, precision, alpha, beta):
        self.joint = GammaGaussian(log_normalizer, info_vec, precision, alpha, beta)

    @staticmethod
    def from_joint(joint):
        return StudentT(joint.log_normalizer, joint.info_vec, joint.precision, joint.alpha, joint.beta)

    def dim(self):
        return self.joint.dim()

    @lazy_property
    def batch_shape(self):
        return self.joint.batch_shape

    def expand(self, batch_shape):
        return StudentT.from_joint(self.joint.expand(batch_shape))

    def reshape(self, batch_shape):
        return StudentT.from_joint(self.joint.reshape(batch_shape))

    def __getitem__(self, index):
        """
        Index into the batch_shape of a StudentT.
        """
        return StudentT.from_joint(self.joint.__getitem__(index))

    @staticmethod
    def cat(parts, dim=0):
        """
        Concatenate a list of StudentTs along a given batch dimension.
        """
        return StudentT.from_joint(GammaGaussian.cat([part.joint for part in parts], dim))

    def event_pad(self, left=0, right=0):
        """
        Pad along event dimension.
        """
        return StudentT.from_joint(self.joint.event_pad(left, right))

    def event_permute(self, perm):
        """
        Permute along event dimension.
        """
        return StudentT.from_joint(self.joint.event_permute(perm))

    def __add__(self, other):
        """
        Approximates the sum of two StudentT in log-density space.
        """
        assert isinstance(other, StudentT)
        assert self.dim() == other.dim()
        new_df = torch.min(self.df, other.df)
        self_st = _absolute_central_moment_matching(self, new_df)
        other_st = _absolute_central_moment_matching(other, new_df)
        return StudentT.from_joint(self_st.joint + other_st.joint)

    def log_density(self, value):
        """
        Evaluate the log density of this StudentT at a point value.

        This is mainly used for testing.
        """
        batch_shape = broadcast_shape(value.shape[:-1], self.batch_shape)
        log_normalizer = self.joint.log_normalizer.expand(batch_shape)
        # compute posterior of mixing variable, then marginalize it
        concentration = self.joint.alpha + 1
        rate = self.joint.beta
        if value.size(-1) > 0:  # nondegenerate case
            P_x = self.joint.precision.matmul(value.unsqueeze(-1)).squeeze(-1)
            rate = rate + ((0.5 * P_x - self.joint.info_vec) * value).sum(-1)
        return Gamma(log_normalizer, concentration, rate).logsumexp()

    def condition(self, value):
        """
        Condition on a trailing subset of its state.
        This should satisfy::

            g.condition(y).dim() == g.dim() - y.size(-1)

        Note that since this is a non-normalized StudentT, we include the
        density of ``y`` in the result. Thus :meth:`condition` is similar to a
        ``functools.partial`` binding of arguments::

            left = x[..., :n]
            right = x[..., n:]
            g.log_density(x) == g.condition(right).log_density(left)
        """
        return StudentT.from_joint(self.joint.condition(value))

    def marginalize(self, left=0, right=0):
        """
        Marginalizing out variables on either side of the event dimension::

            g.marginalize(left=n).event_logsumexp() = g.event_logsumexp()
            g.marginalize(right=n).event_logsumexp() = g.event_logsumexp()

        and for data ``x``:

            g.condition(x).event_logsumexp()
              = g.marginalize(left=g.dim() - x.size(-1)).log_density(x)
        """
        return StudentT.from_joint(self.joint.marginalize(left, right))

    def event_logsumexp(self):
        """
        Integrates out all latent state (i.e. operating on event dimensions).
        """
        return self.joint.event_logsumexp().logsumexp()

    def to_mvt(self):
        """
        Returns the corresponding MultivariateStudentT distribution, ignores log normalizer.
        """
        return self.joint.compound()


def mvt_to_studentt(mvt):
    """
    Convert a MultivariateStudentT distribution to a StudentT.

    :param ~pyro.distributions.MultivariateStudentT mvt: A multivariate student-t distribution.
    :return: An equivalent StudentT object.
    :rtype: ~pyro.ops.studentt.StudentT
    """
    assert isinstance(mvt, pyro.distributions.MultivariateStudentT)
    half_df = 0.5 * mvt.df
    gamma = pyro.distributions.Gamma(half_df, half_df)
    mvn = pyro.distributions.MultivariateNormal(mvt.loc, scale_tril=mvt.scale_tril)
    return StudentT.from_joint(gamma_and_mvn_to_gamma_gaussian(gamma, mvn))


def matrix_and_mvt_to_studentt(matrix, mvt):
    """
    Convert a noisy affine function to a StudentT.

        y = x @ matrix + mvt.sample()

    :param ~torch.Tensor matrix: A matrix with rightmost shape ``(x_dim, y_dim)``.
    :param ~pyro.distributions.MultivariateStudentT mvt: A multivariate student-t distribution.
    :return: A StudentT with broadcasted batch shape and ``.dim() == x_dim + y_dim``.
    :rtype: ~pyro.ops.studentt.StudentT
    """
    assert isinstance(mvt, pyro.distributions.MultivariateStudentT)
    assert isinstance(matrix, torch.Tensor)
    x_dim, y_dim = matrix.shape[-2:]
    assert mvt.event_shape == (y_dim,)
    batch_shape = broadcast_shape(matrix.shape[:-2], mvt.batch_shape)
    matrix = matrix.expand(batch_shape + (x_dim, y_dim))
    mvt = mvt.expand(batch_shape)

    y_studentt = mvt_to_studentt(mvt)
    P_yy = y_studentt.joint.precision
    neg_P_xy = matrix.matmul(P_yy)
    P_xy = -neg_P_xy
    P_yx = P_xy.transpose(-1, -2)
    P_xx = neg_P_xy.matmul(matrix.transpose(-1, -2))
    precision = torch.cat([torch.cat([P_xx, P_xy], -1),
                           torch.cat([P_yx, P_yy], -1)], -2)
    info_y = y_studentt.joint.info_vec
    info_x = -matrix.matmul(info_y.unsqueeze(-1)).squeeze(-1)
    info_vec = torch.cat([info_x, info_y], -1)
    log_normalizer = y_studentt.joint.log_normalizer
    alpha = y_studentt.joint.alpha
    beta = y_studentt.joint.beta

    result = StudentT(log_normalizer, info_vec, precision, alpha, beta)
    assert result.batch_shape == batch_shape
    assert result.dim() == x_dim + y_dim
    return result


def studentt_tensordot(x, y, dims=0):
    """
    Computes the integral over two StudentT:

        `(x @ y)(a,c) = log(integral(exp(x(a,b) + y(b,c)), b))`,

    where `x` is a student-t over variables (a,b), `y` is a student-t over variables
    (b,c), (a,b,c) can each be sets of zero or more variables, and `dims` is the size of b.

    :param x: a StudentT instance
    :param y: a StudentT instance
    :param dims: number of variables to contract
    """
    assert isinstance(x, StudentT)
    assert isinstance(y, StudentT)
    na = x.dim() - dims
    nb = dims
    nc = y.dim() - dims
    assert na >= 0
    assert nb >= 0
    assert nc >= 0

    device = x.joint.info_vec.device
    perm = torch.cat([
        torch.arange(na, device=device),
        torch.arange(x.dim(), x.dim() + nc, device=device),
        torch.arange(na, x.dim(), device=device)])
    return (x.event_pad(right=nc) + y.event_pad(left=na)).event_permute(perm).marginalize(right=nb)
