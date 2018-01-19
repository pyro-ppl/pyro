from __future__ import absolute_import, division, print_function

import math

import torch
from torch.autograd import Function

from pyro.distributions.distribution import Distribution
from pyro.distributions.util import copy_docs_from


def _linear_solve_compat(matrix, matrix_chol, y):
    """Solves the equation ``torch.mm(matrix, x) = y`` for x."""
    if matrix.requires_grad or y.requires_grad:
        # If derivatives are required, use the more expensive gesv.
        return torch.gesv(y, matrix)[0]
    else:
        # Use the cheaper Cholesky solver.
        return torch.potrs(y, matrix_chol)


def _matrix_inverse_compat(matrix, matrix_chol):
    """Computes the inverse of a positive semidefinite square matrix."""
    if matrix.requires_grad or matrix_chol.requires_grad:
        # If derivatives are required, use the more expensive inverse.
        return torch.inverse(matrix)
    else:
        # Use the cheaper Cholesky based potri.
        return torch.potri(matrix_chol)


class _NormalizationConstant(Function):
    """
    This computes either zero or the true normalization constant depending on normalized,
    but always computes the true gradient.
    """
    @staticmethod
    def forward(ctx, covariance_matrix, scale_tril, inverse, dimension, normalized):
        ctx.save_for_backward(inverse, scale_tril)
        if not normalized:
            return scale_tril.new([0])
        return scale_tril.diag().log().sum(-1) + (dimension / 2) * math.log(2 * math.pi)

    @staticmethod
    def backward(ctx, grad_output):
        inverse, scale_tril = ctx.saved_variables
        grad = inverse - 0.5 * torch.diag(torch.diag(inverse))
        grad_cholesky = torch.diag(torch.diag(torch.inverse(scale_tril)))
        return grad_output * grad, grad_output * grad_cholesky, None, None, None


@copy_docs_from(Distribution)
class MultivariateNormal(Distribution):
    """Multivariate normal (Gaussian) distribution.

    A distribution over vectors in which all the elements have a joint Gaussian
    density.

    :param torch.autograd.Variable loc: Mean. Must be a vector (Variable
        containing a 1d Tensor).
    :param torch.autograd.Variable covariance_matrix: Covariance matrix.
        Must be symmetric and positive semidefinite.
    :param torch.autograd.Variable scale_tril: The Cholesky decomposition of
        the covariance matrix in lower-triangular form.
    :param torch.autograd.Variable scale_triu: The Cholesky decomposition of
        the covariance matrix in upper-triangular form.
    :param normalized: If set to `False` the normalization constant is omitted
        in the results of batch_log_pdf and log_pdf. This might be preferable,
        as computing the determinant of the covariance matrix might not always
        be numerically stable. Defaults to `True`.
    """
    reparameterized = True

    def __init__(self, loc, covariance_matrix=None, scale_tril=None, scale_triu=None,
                 batch_size=None, normalized=True):
        super(MultivariateNormal, self).__init__()
        if sum([covariance_matrix is None, scale_tril is None, scale_triu is None]) != 2:
            raise ValueError('Exactly one of covariance_matrix, scale_tril, scale_triu must be specified')
        if loc.dim() > 1:
            raise ValueError("The mean must be a vector, but got loc.size() = {}".format(loc.size()))
        if covariance_matrix is not None:
            if covariance_matrix.dim() != 2:
                raise ValueError("The covariance matrix must be a matrix, but got covariance_matrix.size() = {}".format(
                    loc.size()))
            scale_triu = torch.potrf(covariance_matrix)
        else:
            if scale_triu is None:
                scale_triu = scale_tril.transpose(-1, -2)
            if scale_triu.dim() != 2:
                raise ValueError("The Cholesky decomposition of the covariance matrix must be a matrix, "
                                 "but got scale_triu.size() = {}".format(loc.size()))
            covariance_matrix = torch.mm(scale_triu.transpose(0, 1), scale_triu)
        self.loc = loc
        self.covariance_matrix = covariance_matrix
        self.scale_triu = scale_triu
        self.output_shape = loc.size()
        self.normalized = normalized
        self.batch_size = batch_size if batch_size is not None else 1

    def batch_shape(self, x=None):
        loc = self.loc.expand(self.batch_size, *self.loc.size()).squeeze(0)
        if x is not None:
            if x.size()[-1] != loc.size()[-1]:
                raise ValueError("The event size for the data and distribution parameters must match.\n"
                                 "Expected x.size()[-1] == self.loc.size()[0], but got {} vs {}".format(
                                     x.size(-1), loc.size(-1)))
            try:
                loc = loc.expand_as(x)
            except RuntimeError as e:
                raise ValueError("Parameter `loc` with shape {} is not broadcastable to "
                                 "the data shape {}. \nError: {}".format(loc.size(), x.size(), str(e)))

        return loc.size()[:-1]

    def event_shape(self):
        return self.loc.size()[-1:]

    def sample(self):
        """Generate a sample with the specified covariance matrix and mean.

        Differentiation wrt. to the covariance matrix is only supported on
        PyTorch version 0.3.0 or higher.
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.sample`

        """
        batch_size = self.batch_size
        white = self.loc.new(batch_size, *self.loc.size()).normal_()
        transformed_sample = self.loc + torch.mm(white, self.scale_triu)
        return transformed_sample if self.reparameterized else transformed_sample.detach()

    def batch_log_pdf(self, x):
        batch_size = x.size()[0] if len(x.size()) > len(self.loc.size()) else 1
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        x = x.view(batch_size, *self.loc.size())
        # TODO It may be useful to switch between _matrix_inverse_compat() and _linear_solve_compat()
        # based on the batch size and the size of the covariance matrix.
        sigma_inverse = _matrix_inverse_compat(self.covariance_matrix, self.scale_triu)
        normalization_factor = _NormalizationConstant.apply(self.covariance_matrix, self.scale_triu, sigma_inverse,
                                                            self.loc.size()[0], self.normalized)
        return -(normalization_factor + 0.5 * torch.sum((x - self.loc).unsqueeze(2) * torch.bmm(
            sigma_inverse.expand(batch_size, *self.scale_triu.size()),
            (x - self.loc).unsqueeze(-1)), 1)).view(batch_log_pdf_shape)

    def analytic_mean(self):
        return self.loc

    def analytic_var(self):
        return torch.diag(self.covariance_matrix)
