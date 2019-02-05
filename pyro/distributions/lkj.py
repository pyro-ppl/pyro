from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import biject_to, constraints, transform_to
from torch.distributions.constraints import Constraint
from torch.distributions.transforms import Transform, ComposeTransform

from pyro.distributions import Beta, TorchDistribution


########################################
# Define constraint
########################################


class _CorrCholesky(Constraint):
    """
    Constrains to lower-triangular square matrices with positive diagonals and
    Euclidean norm of each row is 1.
    """
    def check(self, value):
        unit_norm_row = ((value.pow(2).sum(-1) - 1).abs() < 1e-6).min(-1)[0]
        return constraints.lower_cholesky.check(value) & unit_norm_row


# TODO rename this public interface to corr_cholesky if move upstream to pytorch
corr_cholesky_constraint = _CorrCholesky()


########################################
# Define transforms
########################################


class _TanhTransform(Transform):
    domain = constraints.real
    codomain = constraints.interval(-1, 1)
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, _TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return ((1 + y) / (1 - y)).log() / 2

    def log_abs_det_jacobian(self, x, y):
        return (1 - y ** 2).log()


class _PartialCorrToCorrCholeskyTransform(Transform):
    """
    Transforms a vector of partial correlations into the cholesky factor of a
    correlation matrix.

    Reference:

    [1] `Cholesky Factors of Correlation Matrices`,
    Stan Reference Manual v2.18, Section 10.12
    """
    domain = constraints.interval(-1, 1)
    codomain = corr_cholesky_constraint
    bijective = True
    sign = +1
    event_dim = 1

    def __eq__(self, other):
        return isinstance(other, _PartialCorrToCorrCholeskyTransform)

    def _call(self, x):
        D = (1.0 + math.sqrt(1.0 + 8.0 * x.size(-1))) / 2.0
        if D % 1 != 0:
            raise ValueError("PartialCorrToCorrCholesky transformation requires d choose 2 inputs.")
        D = int(D)

        y = x.new_zeros(x.shape[:-1] + (D, D))
        y[..., 0, 0] = 1
        y[..., 1:, 0] = x[..., :(D - 1)]

        pos_x = D - 1
        past_y_squared_sum = x.new_zeros(x.shape[:-1] + (D,))
        # FIX ME: find a vectorized way to compute y instead of loop
        for j in range(1, D):
            # XXX we need `.clone()` operator here to avoid the in-place operator error
            # in the backward pass
            past_y_squared_sum = past_y_squared_sum[..., 1:] + y[..., j:, (j - 1)].clone() ** 2
            y[..., j, j] = (1 - past_y_squared_sum[..., 0]).sqrt()
            new_pos_x = pos_x + D - 1 - j
            y[..., (j + 1):, j] = x[..., pos_x:new_pos_x] * (1 - past_y_squared_sum[..., 1:]).sqrt()
            pos_x = new_pos_x
        return y

    def _inverse(self, y):
        x = y.new_ones(y.shape)
        triu_index = x.triu(diagonal=1) > 0.5
        x[..., :, 0] = y[..., :, 0]
        x[..., :, 1:] = y[..., :, 1:] / (1 - y.pow(2).cumsum(-1)[..., :, :-1]).sqrt()
        # we transpose and take upper triangular indices to arrange the result vector
        # by (x21, x31, x41,..., x32, x42,...) instead of (x21, x31, x32, x41, x42,...)
        return x.transpose(-1, -2)[triu_index].reshape(x.shape[:-2] + (-1,))

    def log_abs_det_jacobian(self, x, y):
        triu_index = y.new_ones(y.shape).triu(diagonal=1) > 0.5
        y_tril_vector = y.transpose(-1, -2)[triu_index].reshape(y.shape[:-2] + (-1,))
        # FIX ME: if necessary, handle the case x = 0 (which does not happen almost surely)
        return (y_tril_vector / x).log().sum(-1)


class CorrCholeskyTransform(ComposeTransform):
    """
    Transforms a real vector into the cholesky factor of a correlation matrix.
    """
    def __init__(self):
        parts = [_TanhTransform(), _PartialCorrToCorrCholeskyTransform()]
        super(CorrCholeskyTransform, self).__init__(parts)


# register transform to global store
@biject_to.register(corr_cholesky_constraint)
@transform_to.register(corr_cholesky_constraint)
def _transform_to_corr_cholesky(constraint):
    return CorrCholeskyTransform()


########################################
# Define distribution
########################################


class LKJCholeskyFactor(TorchDistribution):
    """
    Generates cholesky factors of correlation matrices using an LKJ prior.

    The expected use is to combine it with a vector of variances and pass it
    to the scale_tril parameter of a multivariate distribution such as MultivariateNormal.

    E.g., if \theta is a (positive) vector of covariances with the same dimensionality
    as this distribution, and \Omega is sampled from this distribution, scale_tril=diag(sqrt(\theta))*\Omega

    Reference:

    [1] `Generating random correlation matrices based on vines and extended onion method`,
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe

    :param int d: Dimensionality of the matrix
    :param torch.Tensor eta: A single positive number parameterizing the distribution.
    """
    arg_constraints = {"eta": constraints.positive}
    support = corr_cholesky_constraint
    has_rsample = True

    def __init__(self, d, eta):
        if not torch.is_tensor(eta):
            eta = torch.FloatTensor([eta])
        if any(eta <= 0):
            raise ValueException("eta must be > 0")
        vector_size = (d * (d - 1)) // 2
        alpha = eta.add(0.5 * (d  - 1.0))

        concentrations = eta.new().expand(vector_size)
        i = 0
        for k in range(d-1):
            alpha -= .5
            for j in range(k+1, d):
                concentrations[i] = alpha
                i += 1
        self._generating_distribution = Beta(concentrations, concentrations)
        self._transformation = CorrCholeskyTransform()
        self._eta = eta
        self._d = d
        self._lkj_constant = None

    def rsample(self):
        return self._transformation(self._generating_distribution.sample().mul(2).add(- 1.0))

    def lkj_constant(self, eta, K):
        if self._lkj_constant is not None:
            return self._lkj_constant

        Km1 = K - 1

        constant = torch.lgamma(eta.add(0.5 * Km1)).mul(Km1)

        k = torch.linspace(start=1, end=Km1, steps=Km1, device=eta.device)
        constant -= (k.mul(math.log(math.pi) * 0.5) + torch.lgamma(eta.add( 0.5 * (Km1 - k)))).sum()

        self._lkj_constant = constant
        return constant

    def log_prob(self, x):
        eta = self._eta

        lp = lkj_constant(eta, x.shape[1])

        Km1 = x.shape[1] - 1
        log_diagonals = x.diag().tail(Km1).log()
        values = torch.linspace(start=Km1 - 1, end=0, steps=km1, device=x.device) * log_diagonals

        values += log_diagonals.mul(eta.mul(2).add(-2.0))

        return values.sum() + lp
