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
        self.X = X
        self.y = y
        self.kernel = kernel
        self.likelihood = likelihood

        self.num_data = self.X.size(0)

        self.jitter = self.X.data.new([jitter])

    def model(self):
        self.set_mode("model")

        kernel = self.kernel
        likelihood = self.likelihood

        zero_loc = self.X.data.new([0]).expand(self.num_data)
        Kff = kernel(self.X) + self.jitter.expand(self.num_data).diag()

        f = pyro.sample("f", dist.MultivariateNormal(zero_loc, Kff))
        likelihood(f, obs=self.y)

    def guide(self):
        self.set_mode("guide")

        kernel = self.kernel
        likelihood = self.likelihood

        # define variational parameters
        mf_0 = torch.tensor(self.X.new(self.num_data).zero_(), requires_grad=True)
        mf = pyro.param("f_loc", mf_0)
        unconstrained_Lf_0 = torch.tensor(self.X.data.new(self.num_data, self.num_data).zero_(),
                                          requires_grad=True)
        unconstrained_Lf = pyro.param("unconstrained_f_tril", unconstrained_Lf_0)
        Lf = transform_to(constraints.lower_cholesky)(unconstrained_Lf)

        # TODO: use scale_tril=Lf
        Kf = Lf.t().matmul(Lf) + self.jitter.expand(self.num_data).diag()
        pyro.sample("f", dist.MultivariateNormal(loc=mf, covariance_matrix=Kf))
        return kernel, likelihood, mf, Lf

    def forward(self, Xnew, full_cov=False):
        """
        Computes the parameters of :math:`p(f^*|Xnew) \sim N(\\text{loc}, \\text{cov})`
        according to :math:`p(f^*,f|y) = p(f^*|f)p(f|y) \sim p(f^*|f)q(f)`,
        then marginalize out variable :math:`f`.

        :param torch.Tensor Xnew: A 1D or 2D tensor.
        :param bool full_cov: Predict full covariance matrix or just its diagonal.
        :return: loc, covariance matrix of :math:`p(f^*|Xnew)`, and the likelihood.
        :rtype: torch.Tensor, torch.Tensor, and
            pyro.contrib.gp.likelihoods.Likelihood
        """
        self._check_Xnew_shape(Xnew, self.X)

        kernel, likelihood, mf, Lf = self.guide()

        loc, cov = self._predict_f(Xnew, self.X, kernel, mf, Lf, full_cov)

        return loc, cov, likelihood

    def _predict_f(self, Xnew, X, kernel, mf, Lf, full_cov=False):
        """
        Computes the parameters of :math:`p(f^*|Xnew) \sim N(\\text{loc}, \\text{cov})`
        according to :math:`p(f^*,f|y) = p(f^*|f)p(f|y) \sim p(f^*|f)q(f)`,
        then marginalize out variable :math:`f`.
        Here :math:`q(f)` is parameterized by :math:`q(f) \sim N(mf, Lf)`.
        """
        num_data = X.size(0)

        # W := inv(Lff) @ Kfs; V := inv(Lff) @ Lf
        # loc = Ksf @ inv(Kff) @ mf = W.T @ inv(Lff) @ mf
        # cov = Kss - Ksf @ inv(Kff) @ Kfs + Ksf @ inv(Kff) @ S @ inv(Kff) @ Kfs
        #     = Kss - W.T @ W + W.T @ V @ V.T @ W
        #     =: Kss - Qss + K

        Kff = kernel(X) + self.jitter.expand(num_data).diag()
        Kfs = kernel(X, Xnew)
        Lff = Kff.potrf(upper=False)

        pack = torch.cat((mf.unsqueeze(1), Kfs, Lf), dim=1)
        Lffinv_pack = matrix_triangular_solve_compat(pack, Lff, upper=False)
        Lffinv_mf = Lffinv_pack[:, 0]
        W = Lffinv_pack[:, 1:-num_data]
        V = Lffinv_pack[:, -num_data:]
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
