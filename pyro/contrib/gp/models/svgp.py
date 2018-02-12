from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable
from torch.distributions import constraints, transform_to
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat

from .model import Model


class SparseVariationalGP(Model):
    """
    Sparse Variational Gaussian Process module.

    References

    [1] `Scalable variational Gaussian process classification`,
    James Hensman, Alexander G. de G. Matthews, Zoubin Ghahramani

    :param torch.autograd.Variable X: A tensor of inputs.
    :param torch.autograd.Variable y: A tensor of outputs for training.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor Xu: An inducing-point parameter.
    :param pyro.contrib.gp.likelihoods.Likelihood likelihood: A likelihood module.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, Xu, likelihood, jitter=1e-6):
        super(SparseVariationalGP, self).__init__()
        self.X = X
        self.y = y
        self.kernel = kernel
        self.num_data = self.X.size(0)

        self.Xu = Parameter(Xu)
        self.num_inducing = self.Xu.size(0)

        self.likelihood = likelihood

        self.jitter = Variable(self.X.data.new([jitter]))

    def model(self):
        kernel = self.kernel.set_mode("model")
        likelihood = self.likelihood.set_mode("model")
        self.set_mode("model")
        Xu = self.get_param("Xu")

        Kuu = kernel(Xu) + self.jitter.expand(self.num_inducing).diag()
        Kuf = kernel(Xu, self.X)
        Luu = Kuu.potrf(upper=False)

        zero_loc = Variable(Xu.data.new([0])).expand(self.num_inducing)
        u = pyro.sample("u", dist.MultivariateNormal(loc=zero_loc, scale_tril=Luu))

        # p(f|u) ~ N(f|mf, Kf)
        # mf = Kfu @ inv(Kuu) @ u; Kf = Kff - Kfu @ inv(Kuu) @ Kuf
        pack = torch.cat((u.unsqueeze(1), Kuf), dim=1)
        Luuinv_pack = matrix_triangular_solve_compat(pack, Luu, upper=False)
        Luuinv_u = Luuinv_pack[:, 0]
        Luuinv_Kuf = Luuinv_pack[:, 1:]

        mf = Luuinv_Kuf.t().matmul(Luuinv_u)
        Kffdiag = kernel(self.X, diag=True)
        Qffdiag = (Luuinv_Kuf ** 2).sum(dim=0)
        Kfdiag = Kffdiag - Qffdiag

        f = dist.Normal(mf, Kfdiag).sample()
        likelihood(f, obs=self.y)

    def guide(self):
        kernel = self.kernel.set_mode("guide")
        likelihood = self.likelihood.set_mode("guide")
        self.set_mode("guide")
        Xu = self.get_param("Xu")

        # define variational parameters
        mu_0 = Variable(Xu.data.new(self.num_inducing).zero_(), requires_grad=True)
        mu = pyro.param("u_loc", mu_0)
        unconstrained_Lu_0 = Variable(Xu.data.new(self.num_inducing, self.num_inducing).zero_(),
                                      requires_grad=True)
        unconstrained_Lu = pyro.param("unconstrained_u_tril", unconstrained_Lu_0)
        Lu = transform_to(constraints.lower_cholesky)(unconstrained_Lu)

        pyro.sample("u", dist.MultivariateNormal(loc=mu, scale_tril=Lu))
        return kernel, likelihood, Xu, mu, Lu

    def forward(self, Xnew, full_cov=False):
        """
        Compute the parameters of ``p(f*|Xnew) ~ N(loc, cov)`` according to
        ``p(f*,u|y) = p(f*|u).p(u|y) ~ p(f*|u).q(u)``, then marginalize out variable ``u``.

        :param torch.autograd.Variable Xnew: A 2D tensor.
        :param bool full_cov: Predict full covariance matrix or just its diagonal.
        :return: loc, covariance matrix of ``p(f*|Xnew)``, and the likelihood.
        :rtype: torch.autograd.Variable, torch.autograd.Variable, and
            pyro.contrib.gp.likelihoods.Likelihood
        """
        if Xnew.dim() == 2 and self.X.size(1) != Xnew.size(1):
            assert ValueError("Train data and test data should have the same feature sizes.")
        if Xnew.dim() == 1:
            Xnew = Xnew.unsqueeze(1)

        kernel, likelihood, Xu, mu, Lu = self.guide()

        # W := inv(Luu) @ Kus; V := inv(Luu) @ Lu
        # loc = Ksu @ inv(Kuu) @ mu = W.T @ inv(Luu) @ mu
        # cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ inv(Kuu) @ S @ inv(Kuu) @ Kus
        #     = Kss - W.T @ W + W.T @ V @ V.T @ W
        #     =: Kss - Qss + K

        Kuu = kernel(Xu) + self.jitter.expand(self.num_inducing).diag()
        Kus = kernel(Xu, Xnew)
        Luu = Kuu.potrf(upper=False)

        # combine all tril-solvers to one place
        pack = torch.cat((mu.unsqueeze(1), Kus, Lu), dim=1)
        Luuinv_pack = matrix_triangular_solve_compat(pack, Luu, upper=False)
        Luuinv_mu = Luuinv_pack[:, 0]
        W = Luuinv_pack[:, 1:-self.num_inducing]
        V = Luuinv_pack[:, -self.num_inducing:]
        Vt_W = V.t().matmul(W)

        loc = W.t().matmul(Luuinv_mu)

        if full_cov:
            Kss = kernel(Xnew)
            # Qss = Ksu @ inv(Kuu) @ Kus = W.T @ W
            Qss = W.t().matmul(W)
            # K = Ksu @ inv(Kuu) @ S @ inv(Kuu) @ Kus = W.T @ V @ V.T @ W
            K = Vt_W.t().matmul(Vt_W)
            cov = Kss - Qss + K
        else:
            Kssdiag = kernel(Xnew, diag=True)
            Qssdiag = (W ** 2).sum(dim=0)
            Kdiag = (Vt_W ** 2).sum(dim=0)
            cov = Kssdiag - Qssdiag + Kdiag

        return loc, cov, likelihood
