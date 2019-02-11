from __future__ import absolute_import, division, print_function

import math
import numbers

import torch
import torch.nn.functional as F
from torch.distributions import biject_to, constraints, transform_to
from torch.distributions.constraints import Constraint
from torch.distributions.transforms import Transform, ComposeTransform

from pyro.distributions.torch import Beta
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import eye_like


########################################
# Define constraint
########################################


class _CorrCholesky(Constraint):
    """
    Constrains to lower-triangular square matrices with positive diagonals and Euclidean norm
    of each row is 1.
    """
    def check(self, value):
        unit_norm_row = ((value.pow(2).sum(-1) - 1).abs() < 1e-6).min(-1)[0]
        return constraints.lower_cholesky.check(value) & unit_norm_row


# TODO rename this public interface to corr_cholesky if move upstream to pytorch
corr_cholesky_constraint = _CorrCholesky()


########################################
# Define transform
########################################


def _signed_stick_breaking_tril(t):
    """
    param torch.Tensor t: a vector in (-1, 1).
    """
    # make sure that x.size(-1) = D * (D - 1) / 2 for some D
    D = round((1 + math.sqrt(1 + 8 * t.size(-1))) / 2)
    if D * (D - 1) != 2 * t.size(-1):
        raise ValueError("This transformation requires an input with last shape is "
                         "D*(D-1)/2 for some positive integer D.")

    # transform t to tril matrix with identity diagonal
    r = t.new_ones(t.shape[:-1] + (D, D)).tril(diagonal=-1)
    tril_index = r > 0.5
    r[tril_index] = t.reshape(-1)
    r.view(r.shape[:-2] + (D * D,))[..., ::D + 1] = 1

    # apply stick-breaking on the squared values;
    # we omit the step of computing s = z * z_cumprod by using the fact:
    #     y = sign(r) * s = sign(r) * sqrt(z * z_cumprod) = r * sqrt(z_cumprod)
    z = r ** 2
    z_cumprod = (1 - z).cumprod(-1)

    # to workaround the issue: NaN propagated through backward pass even when not accessed
    # at https://github.com/pytorch/pytorch/issues/15506,
    # here we only take sqrt at tril_index
    z_cumprod_sqrt = z_cumprod.new_zeros(z_cumprod.shape)
    z_cumprod_sqrt[tril_index] = z_cumprod[tril_index].sqrt()
    z_cumprod_sqrt_shifted = F.pad(z_cumprod_sqrt[..., :-1], pad=(1, 0), value=1)
    y = r * z_cumprod_sqrt_shifted
    return y


class CorrCholeskyTransform(Transform):
    """
    Transforms a uncontrained real vector `x` with length `D*(D-1)/2` into the Cholesky factor
    of a D-dimension correlation matrix. This Cholesky factor is a lower triangular matrix
    with positive diagonals and unit Euclidean norm for each row.

    The transform is processed as follows:

    1. First we convert a `x` into a lower triangular matrix with the following order:
        \begin{bmatrix}
            1   & 0 & 0 & 0 \\
            x_0 & 1 & 0 & 0 \\
            x_1 & x_2 & 1 & 0 \\
            x_3 & x_4 & x_5 & 1
        \end{bmatrix}
    2. For each row `X_i` of the lower triangular part, we apply a *signed* version of class
    :class:`~torch.distributions.StickBreakingTransform` to transform `X_i` into a unit
    Euclidean length vector using the following steps:
        a. Scales into the interval `(-1, 1)` domain: defines :math:`r_i = \tanh(X_i)`.
        b. Transforms into an unsigned domain: defines :math:`z_i = r_i^2`.
        c. Applies `s_i = StickBreakingTransform(z_i)`. Note that `s_i` has an additional
        dimension.
        d. Transforms back into signed domain: returns
        :math:`y_i = (sign(r_i), 1) * \sqrt{s_i}`.
    """
    domain = constraints.real
    codomain = corr_cholesky_constraint
    bijective = True
    sign = +1
    event_dim = 1

    def __eq__(self, other):
        return isinstance(other, _PartialCorrToCorrCholeskyTransform)

    def _call(self, x):
        # we interchange step 1 and step 2.a for a better performance
        eps = torch.finfo(x.dtype).eps
        t = x.tanh().clamp(min=(-1 + eps), max=(1 - eps))

        return _signed_stick_breaking_tril(t)

    def _inverse(self, y):
        # inverse stick-breaking
        z_cumprod = 1 - y.pow(2).cumsum(-1)
        z_cumprod_shifted = F.pad(z_cumprod[..., :-1], pad=(1, 0), value=1)
        tril_index = y.new_ones(y.shape).tril(diagonal=-1) > 0.5
        t = y[tril_index] / z_cumprod_shifted[tril_index].sqrt()
        # inverse of tanh
        x = ((1 + t) / (1 - t)).log() / 2
        return x.reshape(y.shape[:-2] + (-1,))

    def log_abs_det_jacobian(self, x, y):
        """
        Because domain and codomain are two spaces with different dimensions, determinant of
        Jacobian is not well-defined. Here we return `log_abs_det_jacobian` of `x` and the
        flatten lower triangular part of `y`.
        """
        # stick_breaking_logdet = log(y / r) = log(z_cumprod)  (modulo right shifted)
        z_cumprod = 1 - y.pow(2).cumsum(-1)
        # by taking diagonal=-2, we don't need to shift z_cumprod to the right
        tril_index = y.new_ones(y.shape).tril(diagonal=-2) > 0.5
        z_cumprod_vector = z_cumprod[tril_index].reshape(y.shape[:-2] + (-1,))
        stick_breaking_logdet = 0.5 * z_cumprod_vector.log().sum(-1)

        tanh_logdet = -2 * x.cosh().log().sum(-1)
        return stick_breaking_logdet + tanh_logdet


# register transform to global store
@biject_to.register(corr_cholesky_constraint)
@transform_to.register(corr_cholesky_constraint)
def _transform_to_corr_cholesky(constraint):
    return CorrCholeskyTransform()


########################################
# Define distribution
########################################


class LKJCorrCholesky(TorchDistribution):
    """
    LKJ distribution for lower Cholesky factors of correlation matrices. The distribution is
    controlled by ``concentration`` parameter :math:`\eta` to make the probability of the
    correlation matrix :math:`M` generated from a Cholesky factor propotional to
    :math:`\det(M)^{\eta - 1}`. Because of that, when ``concentration == 1``, we have a
    uniform distribution over Cholesky factors of correlation matrices.

    When ``concentration > 1``, the distribution favors samples with large diagonal entries
    (hence large determinent). This is useful when we know a priori that the underlying
    variables are not correlated.

    When ``concentration < 1``, the distribution favors samples with small diagonal entries
    (hence small determinent). This is useful when we know a priori that some underlying
    variables are correlated.

    Reference:

    [1] `Generating random correlation matrices based on vines and extended onion method`,
    Daniel Lewandowski, Dorota Kurowicka, Harry Joe

    :param int dimension: dimension of the matrices
    :param float or torch.Tensor concentration: concentration/shape parameter of the
        distribution (often referred to as eta)
    :param str sample_method: Either "cvine" or "onion". Both methods are proposed in [1] and 
        offer the same distribution over correlation matrices. But they are different in how
        to generate samples. Defaults to "onion".
    """
    arg_constraints = {"concentration": constraints.positive}
    support = corr_cholesky_constraint
    has_rsample = True

    def __init__(self, dimension, concentration, sample_method="onion", validate_args=None):
        if dimension < 2:
            raise ValueError("Dimension must be greater than or equal to 2.")
        self.dimension = dimension
        if isinstance(concentration, numbers.Number):
            concentration = torch.tensor(float(concentration))
        self.concentration = concentration
        batch_shape = concentration.shape
        event_shape = torch.Size([dimension, dimension])

        N = dimension * (dimension - 1) // 2
        # We construct base distributions to generate samples for each method.
        # The purpose of this base distribution is to generate a distribution for
        # correlation matrices which is propotional to `det(M)^{\eta - 1}`.
        # (note that this is not a unique way to define base distribution)
        # Both of the following methods have marginal distribution of each off-diagonal
        # element of sampled correlation matrices is Beta(eta + (D-2) / 2, eta + (D-2) / 2).
        if sample_method == "cvine":
            # The following construction follows from the algorithm in Section 2.4 of [1].
            beta_concentration_init = concentration + (dimension - 1) / 2.
            beta_concentration_offset = torch.arange(0.5, dimension / 2. - 0.1, step=0.5,
                                                     dtype=concentration.dtype,
                                                     device=concentration.device)
            beta_concentration = (beta_concentration_init.unsqueeze(-1) -
                                  beta_concentration_offset)
            # expand to a matrix then takes the vector form of the lower triangular part;
            # here we unsqueeze to make the generated vector be in a correct order.
            beta_concentration = beta_concentration.unsqueeze(-2).expand(
                batch_shape + (dimension - 1, dimension - 1))
            tril_index = concentration.new_ones(beta_concentration.shape).tril() > 0.5
            beta_concentration_vector = beta_concentration[tril_index].reshape(
                batch_shape + (N,))
            self._beta_dist = Beta(beta_concentration_vector, beta_concentration_vector)
        elif sample_method == "onion":
            # The following construction follows from the algorithm in Section 3.2 of [1].
            beta_concentration0_init = concentration + (dimension - 1) / 2.
            beta_concentration0_offset = torch.arange(0.5, dimension / 2. - 0.1, step=0.5,
                                                      dtype=concentration.dtype,
                                                      device=concentration.device)
            beta_concentration0 = (beta_concentration0_init.unsqueeze(-1) -
                                   beta_concentration0_offset)
            beta_concentration1 = beta_concentration0_offset
            # expand to a matrix then takes the vector form of the lower triangular part;
            # here we unsqueeze to make the generated vector be in a correct order.
            beta_concentration0 = beta_concentration0.unsqueeze(-1).expand(
                batch_shape + (dimension - 1, dimension - 1))
            beta_concentration1 = beta_concentration1.unsqueeze(-1).expand(
                batch_shape + (dimension - 1, dimension - 1))
            tril_index = concentration.new_ones(beta_concentration0.shape).tril() > 0.5
            beta_concentration0_vector = beta_concentration0[tril_index].reshape(
                batch_shape + (N,))
            beta_concentration1_vector = beta_concentration1[tril_index].reshape(
                batch_shape + (N,))
            self._beta_dist = Beta(beta_concentration1_vector, beta_concentration0_vector)
        else:
            raise ValueError("`method` should be one of 'cvine' or 'onion'.")
        self.sample_method = sample_method

        super(LKJCorrCholesky, self).__init__(batch_shape, event_shape,
                                              validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LKJCorrCholesky, _instance)
        batch_shape = torch.Size(batch_shape)
        new.dimension = self.dimension
        new.concentration = self.concentration.expand(batch_shape)
        new.sample_method = self.sample_method
        new._beta_dist = self._beta_dist.expand(batch_shape +
                                                self._beta_dist.batch_shape[-1:])
        super(LKJCorrCholesky, new).__init__(batch_shape, self.event_shape,
                                             validate_args=False)
        new._validate_args = self._validate_args
        return new

    def log_prob(self, value):
        # Note about computing Jacobain of the transformation from Cholesky factor to
        # correlation matrix:
        #
        #   Assume C = L@Lt and L = (1 0 0; a \sqrt(1-a^2) 0; b c \sqrt(1-b^2-c^2)), we have
        #   Then off-diagonal lower triangular vector of L is transformed to the off-diagonal
        #   lower triangular vector of C by the transform:
        #       (a, b, c) -> (a, b, ab + c\sqrt(1-a^2))
        #   Hence, Jacobian = 1 * 1 * \sqrt(1 - a^2) = \sqrt(1 - a^2) = L22, where L22
        #       is the 2th diagonal element of L
        #   Generally, for a D dimensional matrix, we have:
        #       Jacobian = L22^(D-2) * L33^(D-3) * ... * Ldd^0
        #
        # From [1], we know that probability of a correlation matrix is propotional to
        #   determinant ** (concentration - 1) = prod(L_ii ^ 2*(concentration - 1))
        # On the other hand, Jabobian of the transformation from Cholesky factor to
        # correlation matrix is:
        #   prod(L_ii ^ (D - i))
        # So the probability of a Cholesky factor is propotional to
        #   prod(L_ii ^ (2 * concentration - 2 + D - i)) =: prod(L_ii ^ order_i)
        # with i = 2..D (we omit the element i = 1 because L_11 = 1)

        # Compute `order` vector (note that we need to reindex i -> i-2):
        order_offset = torch.arange(4 - self.dimension, 2.1, dtype=self.concentration.dtype,
                                    device=self.concentration.device)
        order = 2 * self.concentration.unsqueeze(-1) - order_offset

        # Compute unnormalized log_prob:
        cholesky_logprob = (order * value.diagonal(dim1=-2, dim2=-1)[..., 1:].log()).sum(-1)

        # Compute normalization constant (on the first proof of page 1999 of [1])
        denominator_concentration = self.concentration + (self.dimension - 1) / 2.
        denominator = torch.lgamma(denominator_concentration)
        numerator = torch.mvlgamma(denominator_concentration - 0.5, self.dimension - 1)
        # pi_constant in [1] is D * (D - 1) / 4
        # pi_constant in torch.mvlgamma is (D - 1) * (D - 2) / 2
        # hence, we need to add a pi_constant = (D - 1) * (1 - D/4)
        pi_constant = (self.dimension - 1) * (1 - self.dimension / 4.) * math.log(math.pi)
        normalization_constant = pi_constant + numerator - denominator

        return cholesky_logprob - normalization_constant

    def rsample(self, sample_shape=torch.Size()):
        if self.sample_method == "cvine":
            return self._rsample_cvine(sample_shape)
        else:
            return self._rsample_onion(sample_shape)

    def _rsample_cvine(self, sample_shape):
        # C-vine method first uses beta_dist to generate partial correlations,
        # then apply signed stick breaking to transform to cholesky factor.
        # FIX ME: I can't find a reference to prove that using signed stick breaking to
        # generate correlation matrices is the same as the C-vine method in [1]. One guarantee
        # is Stan devs follows this approach. Here is an attempt to prove the correctness for
        # the entry r_32.
        #
        # With notations follow from [1], we define p: partial correlation matrix,
        # c: cholesky factor, r: correlation matrix.
        #
        # From recursive formula (2) in [1], we have
        #   r_32 = p_32 * sqrt{(1 - p_21^2)*(1 - p_31^2)} + p_21 * p_31 =: I
        # On the other hand, signed stick breaking process gives:
        #   l_21 = p_21, l_31 = p_31, l_22 = sqrt(1 - p_21^2), l_32 = p_32 * sqrt(1 - p_31^2)
        #   r_32 = l_21 * l_31 + l_22 * l_32
        #        = p_21 * p_31 + p_32 * sqrt{(1 - p_21^2)*(1 - p_31^2)} = I
        beta_sample = self._beta_dist.rsample(sample_shape)
        partial_correlation = 2 * beta_sample - 1
        return _signed_stick_breaking_tril(partial_correlation)

    def _rsample_onion(self, sample_shape):
        beta_sample = self._beta_dist.rsample(sample_shape)
        D = self.dimension

        # Now we generate w term in Algorithm 3.2 of [1].
        # The following Normal distribution is used to create a uniform distribution on
        # a hypershere (ref: http://mathworld.wolfram.com/HyperspherePointPicking.html)
        loc = beta_sample.new_zeros(beta_sample.shape[:-1] + (D - 1, D - 1))
        scale = loc.new_ones(loc.shape)
        normal_sample = torch.normal(loc, scale).tril()
        sphere_uniform_sample = normal_sample / normal_sample.norm(dim=-1, keepdim=True)
        tril_index = scale.tril() > 0.5
        w = beta_sample.sqrt().reshape(-1) * sphere_uniform_sample[tril_index]

        # Note that w is the triangular part of a Cholesky factor of a correlation
        # matrix (from the procedure in algorithm 3.2 of [1]).
        # The diagonal entries of Cholesky factor is sqrt(1 - w^2). We can show it by linear
        # algebra or by recalling that each row of Cholesky factor has unit Euclidean length.
        cholesky = beta_sample.new_zeros(beta_sample.shape[:-1] + (D, D))
        tril_index = cholesky.new_ones(cholesky.shape).tril(diagonal=-1) > 0.5
        cholesky[tril_index] = w
        cholesky_diag = (1 - cholesky.pow(2).sum(-1)).sqrt()
        cholesky.view(cholesky.shape[:-2] + (D * D,))[..., ::D + 1] = cholesky_diag
        return cholesky
