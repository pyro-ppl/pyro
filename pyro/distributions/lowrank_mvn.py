from __future__ import absolute_import, division, print_function

import math

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import matrix_triangular_solve_compat


class LowRankMultivariateNormal(TorchDistribution):
    """
    Low Rank Multivariate Normal distribution.

    Implements fast computation for log probability of Multivariate Normal distribution
    when the covariance matrix has the form::

        covariance_matrix = W.T @ W + D.

    Here D is a diagonal vector and ``W`` is a matrix of size ``M x N``. The
    computation will be beneficial when ``M << N``.

    :param torch.Tensor loc: Mean.
        Must be a 1D or 2D tensor with the last dimension of size N.
    :param torch.Tensor W_term: W term of covariance matrix.
        Must be in 2 dimensional of size M x N.
    :param torch.Tensor D_term: D term of covariance matrix.
        Must be in 1 dimensional of size N.
    :param float trace_term: A optional term to be added into Mahalabonis term
        according to p(y) = N(y|loc, cov).exp(-1/2 * trace_term).
    """
    arg_constraints = {"loc": constraints.real,
                       "covariance_matrix_D_term": constraints.positive,
                       "scale_tril": constraints.lower_triangular}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, W_term, D_term, trace_term=None):
        if loc.shape[-1] != D_term.shape[0]:
            raise ValueError("Expected loc.shape == D_term.shape, but got {} vs {}".format(
                loc.shape, D_term.shape))
        if D_term.shape[0] != W_term.shape[1]:
            raise ValueError("The dimension of D_term must match the second dimension of W_term.")
        if D_term.dim() != 1 or W_term.dim() != 2 or loc.dim() > 2:
            raise ValueError("D_term, W_term must be 1D, 2D tensors respectively and "
                             "loc must be a 1D or 2D tensor.")

        self.loc = loc
        self.covariance_matrix_D_term = D_term
        self.covariance_matrix_W_term = W_term
        self.trace_term = trace_term if trace_term is not None else 0

        batch_shape, event_shape = loc.shape[:-1], loc.shape[-1:]
        super(LowRankMultivariateNormal, self).__init__(batch_shape, event_shape)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.covariance_matrix_D_term + (self.covariance_matrix_W_term ** 2).sum(0)

    @lazy_property
    def scale_tril(self):
        # We use the following formula to increase the numerically computation stability
        # when using Cholesky decomposition (see GPML section 3.4.3):
        #     D + W.T @ W = D1/2 @ (I + D-1/2 @ W.T @ W @ D-1/2) @ D1/2
        Dsqrt = self.covariance_matrix_D_term.sqrt()
        A = self.covariance_matrix_W_term / Dsqrt
        At_A = A.t().matmul(A)
        N = A.shape[1]
        Id = torch.eye(N, N, out=A.new_empty(N, N))
        K = Id + At_A
        L = K.potrf(upper=False)
        return Dsqrt.unsqueeze(1) * L

    def rsample(self, sample_shape=torch.Size()):
        white = self.loc.new_empty(sample_shape + self.loc.shape).normal_()
        return self.loc + torch.matmul(white, self.scale_tril.t())

    def log_prob(self, value):
        delta = value - self.loc
        logdet, mahalanobis_squared = self._compute_logdet_and_mahalanobis(
            self.covariance_matrix_D_term, self.covariance_matrix_W_term, delta, self.trace_term)
        normalization_const = 0.5 * (self.event_shape[-1] * math.log(2 * math.pi) + logdet)
        return -(normalization_const + 0.5 * mahalanobis_squared)

    def _compute_logdet_and_mahalanobis(self, D, W, y, trace_term=0):
        """
        Calculates log determinant and (squared) Mahalanobis term of covariance
        matrix ``(D + Wt.W)``, where ``D`` is a diagonal matrix, based on the
        "Woodbury matrix identity" and "matrix determinant lemma"::

            inv(D + Wt.W) = inv(D) - inv(D).Wt.inv(I + W.inv(D).Wt).W.inv(D)
            log|D + Wt.W| = log|Id + Wt.inv(D).W| + log|D|
        """
        W_Dinv = W / D
        M = W.shape[0]
        Id = torch.eye(M, M, out=W.new_empty(M, M))
        K = Id + W_Dinv.matmul(W.t())
        L = K.potrf(upper=False)
        if y.dim() == 1:
            W_Dinv_y = W_Dinv.matmul(y)
        elif y.dim() == 2:
            W_Dinv_y = W_Dinv.matmul(y.t())
        else:
            raise NotImplementedError("SparseMultivariateNormal distribution does not support "
                                      "computing log_prob for a tensor with more than 2 dimensionals.")
        Linv_W_Dinv_y = matrix_triangular_solve_compat(W_Dinv_y, L, upper=False)
        if y.dim() == 2:
            Linv_W_Dinv_y = Linv_W_Dinv_y.t()

        logdet = 2 * L.diag().log().sum() + D.log().sum()

        mahalanobis1 = (y * y / D).sum(-1)
        mahalanobis2 = (Linv_W_Dinv_y * Linv_W_Dinv_y).sum(-1)
        mahalanobis_squared = mahalanobis1 - mahalanobis2 + trace_term

        return logdet, mahalanobis_squared
