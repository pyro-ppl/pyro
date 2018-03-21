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
    :param torch.Tensor y: A tensor of output data for training with
        ``y.size(0)`` equals to number of data points.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor Xu: Initial values for inducing points, which are parameters
        of our model.
    :param pyro.contrib.gp.likelihoods.Likelihood likelihood: A likelihood module.
    :param torch.Size latent_shape: Shape for latent processes. By default, it equals
        to output batch shape ``y.size()[1:]``. For the multi-class classification
        problems, ``latent_shape[-1]`` should corresponse to the number of classes
        to predict.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, Xu, likelihood, latent_shape=None, jitter=1e-6):
        super(SparseVariationalGP, self).__init__(X, y, kernel, likelihood,
                                                  latent_shape, jitter)
        self.Xu = Parameter(Xu)

    def model(self):
        self.set_mode("model")

        kernel = self.kernel
        likelihood = self.likelihood
        Xu = self.get_param("Xu")

        Kuu = kernel(Xu) + self.jitter.expand(Xu.size(0)).diag()
        Kuf = kernel(Xu, self.X)
        Luu = Kuu.potrf(upper=False)

        mu_shape = self.latent_shape + Xu.size()[:1]
        zero_loc = Xu.new([0]).expand(mu_shape)
        u = pyro.sample("u", dist.MultivariateNormal(loc=zero_loc, scale_tril=Luu))

        # p(f|u) ~ N(f|mf, Kf)
        # mf = Kfu @ inv(Kuu) @ u; Kf = Kff - Kfu @ inv(Kuu) @ Kuf = Kff - W.T @ W
        # convert u_shape from latent_shape x N to N x latent_shape
        u = u.permute(-1, *range(u.dim())[:-1])
        # convert u to 2D tensors before packing
        u_temp = u.view(u.size(0), -1)
        pack = torch.cat((u_temp, Kuf), dim=1)
        Luuinv_pack = matrix_triangular_solve_compat(pack, Luu, upper=False)
        # unpack
        Luuinv_u = Luuinv_pack[:, :u_temp.size(1)]
        W = Luuinv_pack[:, u_temp.size(1):]

        mf_shape = self.X.size()[:1] + self.latent_shape
        mf = W.t().matmul(Luuinv_u).view(mf_shape)
        # convert mf_shape from N x latent_shape to latent_shape x N
        mf = mf.permute(*range(mf.dim())[1:], 0)
        Kffdiag = kernel(self.X, diag=True)
        Qffdiag = (W ** 2).sum(dim=0)
        Kfdiag = Kffdiag - Qffdiag

        # get 1 sample for f
        f = dist.Normal(mf, Kfdiag)()
        # convert y_shape from N x D to D x N
        y = self.y.permute(*range(self.y.dim())[1:], 0)
        likelihood(f, obs=y)

    def guide(self):
        self.set_mode("guide")

        kernel = self.kernel
        likelihood = self.likelihood
        Xu = self.get_param("Xu")

        # define variational parameters
        mu_shape = self.latent_shape + Xu.size()[:1]
        mu_0 = torch.tensor(Xu.data.new(mu_shape).zero_(),
                            requires_grad=True)
        mu = pyro.param("u_loc", mu_0)
        Lu_shape = self.latent_shape + torch.Size([Xu.size(0), Xu.size(0)])
        # TODO: use new syntax for pyro.param constraint
        unconstrained_Lu_0 = torch.tensor(Xu.new(Lu_shape).zero_(),
                                          requires_grad=True)
        unconstrained_Lu = pyro.param("unconstrained_u_tril", unconstrained_Lu_0)
        Lu = transform_to(constraints.lower_cholesky)(unconstrained_Lu)

        pyro.sample("u", dist.MultivariateNormal(loc=mu, scale_tril=Lu))
        return kernel, Xu, likelihood, mu, Lu

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

        kernel, Xu, likelihood, mu, Lu = self.guide()

        loc, cov = self._predict_f(Xnew, Xu, kernel, mu, Lu, full_cov)

        return loc, cov
