from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints, transform_to
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat

from .vgp import VariationalGP


class SparseVariationalGP(VariationalGP):
    """
    Sparse Variational Gaussian Process module.

    References

    [1] `Scalable variational Gaussian process classification`,
    James Hensman, Alexander G. de G. Matthews, Zoubin Ghahramani

    :param torch.Tensor X: A 1D or 2D tensor of input data for training.
    :param torch.Tensor y: A 1D or 2D tensor of output data for training.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor Xu: Initial values for inducing points, which are parameters
        of our model.
    :param pyro.contrib.gp.likelihoods.Likelihood likelihood: A likelihood module.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, Xu, likelihood, jitter=1e-6):
        super(SparseVariationalGP, self).__init__(X, y, kernel, likelihood, jitter)
        self.Xu = Parameter(Xu)

    def model(self):
        self.set_mode("model")

        kernel = self.kernel
        likelihood = self.likelihood
        Xu = self.get_param("Xu")

        Kuu = kernel(Xu) + self.jitter.expand(Xu.size(0)).diag()
        Kuf = kernel(Xu, self.X)
        Luu = Kuu.potrf(upper=False)

        if self.y.dim() == 1:
            zero_loc = Xu.new([0]).expand(Xu.size(0))
        else:
            zero_loc = Xu.new([0]).expand(self.y.size(1), Xu.size(0))
        # TODO: use scale_tril=Luu
        u = pyro.sample("u", dist.MultivariateNormal(loc=zero_loc, covariance_matrix=Kuu))

        # p(f|u) ~ N(f|mf, Kf)
        # mf = Kfu @ inv(Kuu) @ u; Kf = Kff - Kfu @ inv(Kuu) @ Kuf
        u = u.unsqueeze(1) if u.dim() == 1 else u.t()
        pack = torch.cat((u, Kuf), dim=1)
        Luuinv_pack = matrix_triangular_solve_compat(pack, Luu, upper=False)
        Luuinv_u = Luuinv_pack[:, :u.size(1)]
        Luuinv_Kuf = Luuinv_pack[:, u.size(1):]

        # correct event_shape for y
        y_t = self.y.t() if self.y.dim() == 2 else self.y
        mf_t = Luuinv_Kuf.t().matmul(Luuinv_u).view(y_t.size())
        Kffdiag = kernel(self.X, diag=True)
        Qffdiag = (Luuinv_Kuf ** 2).sum(dim=0)
        Kfdiag = Kffdiag - Qffdiag

        # get 1 sample for f
        f = dist.Normal(mf_t, Kfdiag)()
        likelihood(f, obs=y_t)

    def guide(self):
        self.set_mode("guide")

        kernel = self.kernel
        likelihood = self.likelihood
        Xu = self.get_param("Xu")

        # define variational parameters
        if self.y.dim() == 1:
            mu_0 = torch.tensor(Xu.data.new(Xu.size(0)).zero_(),
                                requires_grad=True)
        else:
            mu_0 = torch.tensor(Xu.data.new(Xu.size(0), self.y.size(1)).zero_(),
                                requires_grad=True)
        mu = pyro.param("u_loc", mu_0)
        unconstrained_Lu_0 = torch.tensor(Xu.new(Xu.size(0), Xu.size(0)).zero_(),
                                          requires_grad=True)
        unconstrained_Lu = pyro.param("unconstrained_u_tril", unconstrained_Lu_0)
        Lu = transform_to(constraints.lower_cholesky)(unconstrained_Lu)

        # TODO: use scale_tril=Lu
        Ku = Lu.t().matmul(Lu) + self.jitter.expand(Lu.size(0)).diag()
        # correct event_shape for mu
        mu_t = mu.t() if mu.dim() == 2 else mu
        pyro.sample("u", dist.MultivariateNormal(loc=mu_t, covariance_matrix=Ku))
        return kernel, likelihood, Xu, mu, Lu

    def forward(self, Xnew, full_cov=False):
        """
        Computes the parameters of :math:`p(f^*|Xnew) \sim N(\\text{loc}, \\text{cov})`
        according to :math:`p(f^*,u|y) = p(f^*|u)p(u|y) \sim p(f^*|u)q(u)`, then
        marginalize out variable :math:`u`. In case output data is a 2D tensor of shape
        :math:`N \times D`, :math:`loc` is also a 2D tensor of shape :math:`N \times D`.
        Covariance matrix :math:`cov` is always a 2D tensor of shape :math:`N \times N`.

        :param torch.Tensor Xnew: A 2D tensor.
        :param bool full_cov: Predict full covariance matrix or just its diagonal.
        :return: loc and covariance matrix of :math:`p(f^*|Xnew)`
        :rtype: torch.Tensor and torch.Tensor
        """
        self._check_Xnew_shape(Xnew, self.X)

        kernel, likelihood, Xu, mu, Lu = self.guide()

        loc, cov = self._predict_f(Xnew, Xu, kernel, mu, Lu, full_cov)

        return loc, cov
