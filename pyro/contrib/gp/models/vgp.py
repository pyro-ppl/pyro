from __future__ import absolute_import, division, print_function

import torch
from torch.autograd import Variable
from torch.distributions import constraints, transform_to
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat


class VariationalGP(nn.Module):
    """
    Variational Gaussian Process module.

    This model can be seen as a special version of SparseVariationalGP model
    with ``Xu = X``.

    :param torch.autograd.Variable X: A tensor of inputs.
    :param torch.autograd.Variable y: A tensor of outputs for training.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param pyro.contrib.gp.likelihoods.Likelihood likelihood: A likelihood module.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, likelihood, kernel_prior=None, jitter=1e-6):
        super(VariationalGP, self).__init__()
        self.X = X
        self.y = y
        self.kernel = kernel
        self.likelihood = likelihood

        self.num_data = self.X.size(0)

        self.jitter = Variable(self.X.data.new([jitter]))

    def model(self):
        kernel = self.kernel.set_mode("model")
        likelihood = self.likelihood.set_mode("model")

        zero_loc = Variable(self.X.data.new([0])).expand(self.num_data)
        Kff = kernel(self.X) + self.jitter.expand(self.num_data).diag()

        f = pyro.sample("f", dist.MultivariateNormal(zero_loc, Kff))
        likelihood(f, obs=self.y)

    def guide(self):
        kernel = self.kernel.set_mode("guide")
        likelihood = self.likelihood.set_mode("guide")

        # define variational parameters
        mf_0 = Variable(self.X.new(self.num_data).zero_(), requires_grad=True)
        mf = pyro.param("f_loc", mf_0)
        unconstrained_Lf_0 = Variable(self.X.data.new(self.num_data, self.num_data).zero_(),
                                      requires_grad=True)
        unconstrained_Lf = pyro.param("unconstrained_f_tril", unconstrained_Lf_0)
        Lf = transform_to(constraints.lower_cholesky)(unconstrained_Lf)

        # TODO: use scale_tril=Lf
        Kf = Lf.t().matmul(Lf) + self.jitter.expand(self.num_data).diag()
        pyro.sample("f", dist.MultivariateNormal(loc=mf, covariance_matrix=Kf))
        return kernel, likelihood, mf, Lf

    def forward(self, Xnew, full_cov=False):
        """
        Compute the parameters of ``p(f*|Xnew) ~ N(loc, cov)`` according to
        ``p(f*,f|y) = p(f*|f).p(f|y) ~ p(f*|f).q(f)``, then marginalize out variable ``f``.

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

        kernel, likelihood, mf, Lf = self.guide()

        # see `SparseVariationalGP` module for the derivation

        Kff = kernel(self.X) + self.jitter.expand(self.num_data)
        Kfs = kernel(self.X, Xnew)
        Lff = Kff.potrf(upper=False)

        pack = torch.cat((mf.unsqueeze(1), Kfs, Lf), dim=1)
        Lffinv_pack = matrix_triangular_solve_compat(pack, Lff, upper=False)
        Lffinv_mf = Lffinv_pack[:, 0]
        W = Lffinv_pack[:, 1:-self.num_data]
        V = Lffinv_pack[:, -self.num_data:]
        Vt_W = V.t().matmul(W)

        loc = W.t().matmul(Lffinv_mf)

        if full_cov:
            Kss = kernel(Xnew)
            Qss = W.t().matmul(W)
            K = Vt_W.t().matmul(Vt_W)
            cov = Kss - Qss + K
        else:
            Kssdiag = kernel(Xnew, diag=True)
            Qssdiag = (W ** 2).sum(dim=0)
            Kdiag = (Vt_W ** 2).sum(dim=0)
            cov = Kssdiag - Qssdiag + Kdiag

        return loc, cov, likelihood
