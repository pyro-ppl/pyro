from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import constraints
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
        problems, ``latent_shape[-1]`` should corresponse to the number of classes.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, Xu, likelihood, latent_shape=None, jitter=1e-6):
        super(SparseVariationalGP, self).__init__(X, y, kernel, likelihood,
                                                  latent_shape, jitter)

        self.Xu = Parameter(Xu)

        num_inducing = self.Xu.shape[0]
        u_loc_shape = self.latent_shape + (num_inducing,)
        u_loc = self.Xu.new_zeros(u_loc_shape)
        self.u_loc = Parameter(u_loc)

        u_scale_tril_shape = self.latent_shape + (num_inducing, num_inducing)
        u_scale_tril = torch.eye(num_inducing, out=self.Xu.new(num_inducing, num_inducing))
        u_scale_tril = u_scale_tril.expand(u_scale_tril_shape)
        self.u_scale_tril = Parameter(u_scale_tril)
        self.set_constraint("u_scale_tril", constraints.lower_cholesky)

    def model(self):
        self.set_mode("model")

        Xu = self.get_param("Xu")

        Kuu = self.kernel(Xu) + self.jitter.expand(Xu.shape[0]).diag()
        Luu = Kuu.potrf(upper=False)
        Kuf = self.kernel(Xu, self.X)

        u_loc_shape = self.latent_shape + (Xu.shape[0],)
        zero_loc = Xu.new_zeros(u_loc_shape)
        u = pyro.sample("u", dist.MultivariateNormal(zero_loc, scale_tril=Luu)
                        .reshape(extra_event_dims=zero_loc.dim()-1))

        # p(f | u) ~ N(f | f_loc, f_cov)
        # f_loc = Kfu @ inv(Kuu) @ u
        # f_cov = Kff - Kfu @ inv(Kuu) @ Kuf = Kff - Qff
        # W = inv(Luu) @ Kuf -> Qff = W.T @ W, f_loc = W.T @ inv(Luu) @ u

        # convert u_shape from latent_shape x N to N x latent_shape
        u = u.permute(-1, *range(u.dim())[:-1]).contiguous()
        # convert u to 2D tensor before packing
        u_temp = u.view(u.shape[0], -1)
        pack = torch.cat((u_temp, Kuf), dim=1)
        Luuinv_pack = matrix_triangular_solve_compat(pack, Luu, upper=False)
        # unpack
        Luuinv_u = Luuinv_pack[:, :u_temp.shape[1]]
        W = Luuinv_pack[:, u_temp.shape[1]:]

        Kffdiag = self.kernel(self.X, diag=True)
        Qffdiag = (W ** 2).sum(dim=0)
        f_var = Kffdiag - Qffdiag

        f_loc_shape = (self.X.shape[0],) + self.latent_shape
        f_loc = W.t().matmul(Luuinv_u).view(f_loc_shape)
        # convert f_loc_shape from N x latent_shape to latent_shape x N
        f_loc = f_loc.permute(list(range(1, f_loc.dim())) + [0]).contiguous()

        # get 1 sample for f
        f = dist.Normal(f_loc, f_var)()

        if self.y is None:
            return self.likelihood(f)
        else:
            # convert y_shape from N x D to D x N
            y = self.y.permute(list(range(1, self.y.dim())) + [0])
            return self.likelihood(f, y)

    def guide(self):
        self.set_mode("guide")

        Xu = self.get_param("Xu")
        u_loc = self.get_param("u_loc")
        u_scale_tril = self.get_param("u_scale_tril")

        pyro.sample("u", dist.MultivariateNormal(loc=u_loc, scale_tril=u_scale_tril)
                    .reshape(extra_event_dims=u_loc.dim()-1))
        return self.kernel, self.likelihood, Xu, u_loc, u_scale_tril

    def forward(self, Xnew, full_cov=False):
        """
        Computes the parameters of :math:`p(f^*|Xnew) \sim N(\\text{loc}, \\text{cov})`
        according to :math:`p(f^*,u|y) = p(f^*|u)p(u|y) \sim p(f^*|u)q(u)`, then
        marginalize out variable :math:`u`. In case output data is a 2D tensor of shape
        :math:`N \times D`, :math:`loc` is also a 2D tensor of shape :math:`N \times D`.
        Covariance matrix :math:`cov` is always a 2D tensor of shape :math:`N \times N`.

        :param torch.Tensor Xnew: A 2D tensor.
        :param bool full_cov: Predict full covariance matrix or just its diagonal.
        :returns: loc and covariance matrix of :math:`p(f^*|Xnew)`
        :rtype: torch.Tensor and torch.Tensor
        """
        self._check_Xnew_shape(Xnew)
        kernel, likelihood, Xu, u_loc, u_scale_tril = self.guide()

        loc, cov = self._predict_f(Xnew, kernel, Xu, u_loc, u_scale_tril, full_cov)
        return loc, cov
