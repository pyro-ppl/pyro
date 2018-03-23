from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat
from pyro.contrib.gp.util import batch_lower_cholesky_transform

from .model import Model


class VariationalGP(Model):
    """
    Variational Gaussian Process module.

    This model can be seen as a special version of SparseVariationalGP model
    with :math:`Xu = X`.

    :param torch.Tensor X: A 1D or 2D tensor of inputs.
    :param torch.Tensor y: A tensor of output data for training with
        ``y.size(0)`` equals to number of data points.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param pyro.contrib.gp.likelihoods.Likelihood likelihood: A likelihood module.
    :param torch.Size latent_shape: Shape for latent processes. By default, it equals
        to output batch shape ``y.size()[1:]``. For the multi-class classification
        problems, ``latent_shape[-1]`` should corresponse to the number of classes.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, likelihood, latent_shape=None, jitter=1e-6):
        super(VariationalGP, self).__init__(X, y, kernel, latent_shape, jitter)
        self.likelihood = likelihood

    def model(self):
        self.set_mode("model")

        kernel = self.kernel
        likelihood = self.likelihood

        mf_shape = self.latent_shape + self.X.size()[:1]
        zero_loc = self.X.new([0]).expand(mf_shape)
        Kff = kernel(self.X) + self.jitter.expand(self.X.size(0)).diag()
        Lff = Kff.potrf(upper=False)

        f = pyro.sample("f", dist.MultivariateNormal(zero_loc, scale_tril=Lff)
                        .reshape(extra_event_dims=zero_loc.dim()-1))
        # convert y_shape from N x D to D x N
        y = self.y.permute(list(range(1, self.y.dim())) + [0])

        likelihood(f, y)

    def guide(self):
        self.set_mode("guide")

        kernel = self.kernel
        likelihood = self.likelihood

        # define variational parameters
        mf_shape = self.latent_shape + self.X.size()[:1]
        mf_0 = torch.tensor(self.X.new(mf_shape).zero_(), requires_grad=True)
        mf = pyro.param("f_loc", mf_0)
        Lf_shape = self.latent_shape + torch.Size([self.X.size(0), self.X.size(0)])
        # TODO: use new syntax for pyro.param constraint
        unconstrained_Lf_0 = torch.tensor(self.X.new(Lf_shape).zero_(), requires_grad=True)
        unconstrained_Lf = pyro.param("unconstrained_f_tril", unconstrained_Lf_0)
        Lf = batch_lower_cholesky_transform(unconstrained_Lf)

        pyro.sample("f", dist.MultivariateNormal(loc=mf, scale_tril=Lf)
                    .reshape(extra_event_dims=mf.dim()-1))
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
        :returns: loc and covariance matrix of :math:`p(f^*|Xnew)`
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

        # convert mf_shape from latent_shape x N to N x latent_shape
        mf = mf.permute(mf.dim()-1, *range(mf.dim())[:-1])
        # convert Lf_shape from latent_shape x N x N to N x N x latent_shape
        Lf = Lf.permute(Lf.dim()-2, Lf.dim()-1, *range(Lf.dim())[:-2]).contiguous()
        # convert mf, Lf to 2D tensors before packing
        mf_temp = mf.view(mf.size(0), -1)
        Lf_temp = Lf.view(Lf.size(0), -1)
        pack = torch.cat((mf_temp, Kfs, Lf_temp), dim=1)
        Lffinv_pack = matrix_triangular_solve_compat(pack, Lff, upper=False)
        # unpack
        Lffinv_mf = Lffinv_pack[:, :mf_temp.size(1)]
        W = Lffinv_pack[:, mf_temp.size(1):-Lf_temp.size(1)]
        V = Lffinv_pack[:, -Lf_temp.size(1):].view(Lf.size())
        # covert V_shape from N x N' x D to D x N' x N
        V_t = V.permute(list(range(2, V.dim())) + [1, 0])
        Vt_W = V_t.matmul(W)

        loc_shape = Xnew.size()[:1] + mf.size()[1:]
        loc = W.t().matmul(Lffinv_mf).view(loc_shape)

        if full_cov:
            Kss = kernel(Xnew)
            # Qss = Ksf @ inv(Kff) @ Kfs = W.T @ W
            Qss = W.t().matmul(W)
            # K = Ksf @ inv(Kff) @ S @ inv(Kff) @ Kfs = W.T @ V @ V.T @ W
            K = Vt_W.transpose(-2, -1).matmul(Vt_W)
            cov = Kss - Qss + K
            # convert cov from D x N x N to N x N x D
            cov = cov.permute(cov.dim()-2, cov.dim()-1, *range(cov.dim())[:-2])
        else:
            Kssdiag = kernel(Xnew, diag=True)
            Qssdiag = (W ** 2).sum(dim=0)
            Kdiag = (Vt_W ** 2).sum(dim=-2)
            cov = Kssdiag - Qssdiag + Kdiag
            # convert cov from D x N to N x D
            cov = cov.permute(cov.dim()-1, *range(cov.dim())[:-1])

        return loc, cov
