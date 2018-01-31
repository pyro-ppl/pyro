from __future__ import absolute_import, division, print_function

import math

import torch
from torch.autograd import Variable
from torch.distributions.utils import lazy_property

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import copy_docs_from, matrix_triangular_solve_compat


@copy_docs_from(Distribution)
class SparseMultivariateNormal(Distribution):
    """
    Sparse Multivariate Normal distribution.

    Implements fast computation for log probability of Multivariate Normal distribution
    when the covariance matrix has the form:
        covariance_matrix = D + W.T @ W.
    Here D is a diagonal vector and W is a matrix of size M x N. The computation will be
    beneficial when M << N.

    :param torch.autograd.Variable loc: Mean.
        Must be in 1 dimensional of size N.
    :param torch.autograd.Variable D_term: D term of covariance matrix.
        Must be in 1 dimensional of size N.
    :param torch.autograd.Variable W_term: W term of covariance matrix.
        Must be in 2 dimensional of size M x N.
    :param float trace_term: A optional term to be added into Mahalabonis term
        according to p(y) = N(y|loc, cov).exp(-1/2 * trace_term).
    """
    reparameterized = True

    def __init__(self, loc, D_term, W_term, trace_term=None, *args, **kwargs):
        if loc.size() != D_term.size():
            raise ValueError("Expected loc.size() == D_term.size(), but got {} vs {}".format(
                loc.size(), D_term.size()))
        if D_term.size(0) != W_term.size(1):
            raise ValueError("The dimension of D_term must match the second dimension of W_term.")
        if loc.dim() != 1 or D_term.dim() != 1 or W_term.dim() != 2:
            raise ValueError("Loc, D_term, W_term must be 1D, 1D, 2D tensors respectively.")

        self.loc = loc
        self.covariance_matrix_D_term = D_term
        self.covariance_matrix_W_term = W_term

        self.trace_term = trace_term if trace_term is not None else 0
        super(SparseMultivariateNormal, self).__init__(*args, **kwargs)
        
    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.covariance_matrix_D_term + (self.covariance_matrix_W_term ** 2).sum(0)

    @lazy_property
    def scale_triu(self):
        # We use the following formula to increase the numerically computation stability
        # when using Cholesky decomposition (see GPML section 3.4.3):
        #     D + W.T @ W = D1/2 @ (I + D-1/2 @ W.T @ W @ D-1/2) @ D1/2
        Dsqrt = self.covariance_matrix_D_term.sqrt()
        A = self.covariance_matrix_W_term / Dsqrt
        At_A = A.t().matmul(A)
        Id = Variable(A.data.new([1])).expand(A.size(1)).diag()
        K = Id + At_A
        U = K.potrf()
        return U * Dsqrt

    def batch_shape(self, x=None):
        event_dim = 1
        loc = self.loc
        if x is not None:
            if x.size()[-event_dim] != loc.size()[-event_dim]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.loc.size()[-1], but got {} vs {}".format(
                                     x.size(-1), loc.size(-1)))
            try:
                loc = self.loc.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `loc` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(loc.size(), x.size(), str(e)))
        return loc.size()[:-event_dim]

    def event_shape(self):
        event_dim = 1
        return self.loc.size()[-event_dim:]

    def sample(self, sample_shape=torch.Size()):
        white = self.loc.new(sample_shape + self.loc.shape).normal_()
        return self.loc + torch.matmul(white, self.scale_triu)

    def log_prob(self, x):
        batch_shape = self.batch_shape(x)
        if batch_shape != torch.Size([]):
            raise ValueError("Batch calculation of log probability is not supported "
                             "for this distribution.")
        delta = x - self.loc
        logdet, mahalanobis_squared = self._compute_logdet_and_mahalanobis(
            self.covariance_matrix_D_term, self.covariance_matrix_W_term, delta, self.trace_term)
        normalization_const = 0.5 * (self.event_shape()[-1] * math.log(2 * math.pi) + logdet)
        return -(normalization_const + 0.5 * mahalanobis_squared)

    def _compute_logdet_and_mahalanobis(self, D, W, y, trace_term=0):
        """
        Calculates log determinant and (squared) Mahalanobis term of covariance
        matrix (D + Wt.W), where D is a diagonal matrix, based on the
        "Woodbury matrix identity" and "matrix determinant lemma":
            inv(D + Wt.W) = inv(D) - inv(D).Wt.inv(I + W.inv(D).Wt).W.inv(D)
            log|D + Wt.W| = log|Id + Wt.inv(D).W| + log|D|
        """
        W_Dinv = W / D
        Id = Variable(W.data.new([1])).expand(W.size(0)).diag()
        K = Id + W_Dinv.matmul(W.t())
        L = K.potrf(upper=False)
        W_Dinv_y = W_Dinv.matmul(y)
        Linv_W_Dinv_y = matrix_triangular_solve_compat(W_Dinv_y, L, upper=False)

        logdet = 2 * L.diag().log().sum() + D.log().sum()

        mahalanobis1 = (y * y / D).sum(-1)
        mahalanobis2 = (Linv_W_Dinv_y * Linv_W_Dinv_y).sum()
        mahalanobis_squared = mahalanobis1 - mahalanobis2 + trace_term

        return logdet, mahalanobis_squared
