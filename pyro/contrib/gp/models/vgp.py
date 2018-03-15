from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints, transform_to

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat

from .model import Model


class VariationalGP(Model):
    """
    Variational Gaussian Process module.

    This model can be seen as a special version of SparseVariationalGP model
    with :math:`Xu = X`.

    :param torch.Tensor X: A 1D or 2D tensor of inputs.
    :param torch.Tensor y: A 1D tensor of outputs for training.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param pyro.contrib.gp.likelihoods.Likelihood likelihood: A likelihood module.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, likelihood, jitter=1e-6):
        super(VariationalGP, self).__init__()
        self.set_data(X, y)
        self.kernel = kernel
        self.likelihood = likelihood

        self.jitter = self.X.data.new([jitter])

    def model(self):
        self.set_mode("model")

        kernel = self.kernel
        likelihood = self.likelihood

        # correct event_shape for y
        y_t = self.y.t() if self.y.dim() == 2 else self.y
        zero_loc = y_t.new([0]).expand(y_t.size())
        Kff = kernel(self.X) + self.jitter.expand(self.X.size(0)).diag()

        f = pyro.sample("f", dist.MultivariateNormal(zero_loc, Kff).reshape(
            extra_event_dims=zero_loc.dim() - 1))
        likelihood(f, obs=y_t)

    def guide(self):
        self.set_mode("guide")

        kernel = self.kernel
        likelihood = self.likelihood

        # define variational parameters
        mf_0 = torch.tensor(self.y.new(self.y.size()).zero_(),
                            requires_grad=True)
        mf = pyro.param("f_loc", mf_0)
        unconstrained_Lf_0 = torch.tensor(self.X.new(self.X.size(0), self.X.size(0)).zero_(),
                                          requires_grad=True)
        unconstrained_Lf = pyro.param("unconstrained_f_tril", unconstrained_Lf_0)
        Lf = transform_to(constraints.lower_cholesky)(unconstrained_Lf)

        # TODO: use scale_tril=Lf
        Kf = Lf.t().matmul(Lf) + self.jitter.expand(Lf.size(0)).diag()
        # correct event_shape for mf
        mf_t = mf.t() if mf.dim() == 2 else mf
        pyro.sample("f", dist.MultivariateNormal(loc=mf_t, covariance_matrix=Kf).reshape(
            extra_event_dims=mf_t.dim() - 1))
        return kernel, likelihood, mf, Lf

    def forward(self, Xnew, full_cov=False):
        """
        Computes the parameters of :math:`p(f^*|Xnew) \sim N(\\text{loc}, \\text{cov})`
        according to :math:`p(f^*,f|y) = p(f^*|f)p(f|y) \sim p(f^*|f)q(f)`, then
        marginalize out variable :math:`f`. In case output data is a 2D tensor of shape
        :math:`N \times D`, :math:`loc` is also a 2D tensor of shape :math:`N \times D`.
        Covariance matrix :math:`cov` is always a 2D tensor of shape :math:`N \times N`.

        :param torch.Tensor Xnew: A 1D or 2D tensor.
        :param bool full_cov: Predict full covariance matrix or just its diagonal.
        :return: loc and covariance matrix of :math:`p(f^*|Xnew)`
        :rtype: torch.Tensor and torch.Tensor
        """
        self._check_Xnew_shape(Xnew, self.X)

        kernel, likelihood, mf, Lf = self.guide()

        loc, cov = self._predict_f(Xnew, self.X, kernel, mf, Lf, full_cov)

        return loc, cov

    def _predict_f(self, Xnew, X, kernel, mf, Lf, full_cov=False):
        """
        Computes the parameters of :math:`p(f^*|Xnew) \sim N(\\text{loc}, \\text{cov})`
        according to :math:`p(f^*,f|y) = p(f^*|f)p(f|y) \sim p(f^*|f)q(f)`,
        then marginalize out variable :math:`f`.
        Here :math:`q(f)` is parameterized by :math:`q(f) \sim N(mf, Lf)`.
        """
        # W := inv(Lff) @ Kfs; V := inv(Lff) @ Lf
        # loc = Ksf @ inv(Kff) @ mf = W.T @ inv(Lff) @ mf
        # cov = Kss - Ksf @ inv(Kff) @ Kfs + Ksf @ inv(Kff) @ S @ inv(Kff) @ Kfs
        #     = Kss - W.T @ W + W.T @ V @ V.T @ W
        #     =: Kss - Qss + K
        Kff = kernel(X) + self.jitter.expand(X.size(0)).diag()
        Kfs = kernel(X, Xnew)
        Lff = Kff.potrf(upper=False)

        mf_temp = mf.unsqueeze(1) if mf.dim() == 1 else mf
        pack = torch.cat((mf_temp, Kfs, Lf), dim=1)
        Lffinv_pack = matrix_triangular_solve_compat(pack, Lff, upper=False)
        Lffinv_mf = Lffinv_pack[:, :mf_temp.size(1)].view(mf.size())
        W = Lffinv_pack[:, mf_temp.size(1):-Lf.size(1)]
        V = Lffinv_pack[:, -Lf.size(1):]
        Vt_W = V.t().matmul(W)

        loc = W.t().matmul(Lffinv_mf)

        if full_cov:
            Kss = kernel(Xnew)
            # Qss = Ksf @ inv(Kff) @ Kfs = W.T @ W
            Qss = W.t().matmul(W)
            # K = Ksf @ inv(Kff) @ S @ inv(Kff) @ Kfs = W.T @ V @ V.T @ W
            K = Vt_W.t().matmul(Vt_W)
            cov = Kss - Qss + K
        else:
            Kssdiag = kernel(Xnew, diag=True)
            Qssdiag = (W ** 2).sum(dim=0)
            Kdiag = (Vt_W ** 2).sum(dim=0)
            cov = Kssdiag - Qssdiag + Kdiag

        return loc, cov
