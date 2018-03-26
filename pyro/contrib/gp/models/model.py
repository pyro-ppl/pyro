from __future__ import absolute_import, division, print_function

import torch

from pyro.contrib.gp.util import Parameterized
from pyro.distributions.util import matrix_triangular_solve_compat
from pyro.infer import SVI
from pyro.optim import Adam, PyroOptim


class GPModel(Parameterized):
    """
    Base class for models used in Gaussian Process.

    :param torch.Tensor X: A 1D or 2D tensor of input data for training.
    :param torch.Tensor y: A tensor of output data for training with
        ``y.shape[-1]`` equals to number of data points.
    :param pyro.contrib.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param torch.Size latent_shape: Shape for latent processes. By default, it equals
        to output batch shape ``y.shape[:-1]``. For the multi-class classification
        problems, ``latent_shape[-1]`` should corresponse to the number of classes.
    :param float jitter: An additional jitter to help stablize Cholesky decomposition.
    """
    def __init__(self, X, y, kernel, latent_shape=None, jitter=1e-6):
        super(GPModel, self).__init__()
        self.set_data(X, y)
        self.kernel = kernel
        y_batch_shape = self.y.shape[:-1]
        self.latent_shape = latent_shape if latent_shape is not None else y_batch_shape
        self.jitter = self.X.new([jitter])

    def set_data(self, X, y=None):
        """
        Sets data for Gaussian Process models.

        :param torch.Tensor X: A 1D or 2D tensor of input data for training.
        :param torch.Tensor y: A tensor of output data for training with
            ``y.shape[-1]`` equals to number of data points.
        """
        if X.dim() > 2:
            raise ValueError("Expected input tensor of 1 or 2 dimensions, "
                             "actual dim = {}".format(X.dim()))
        if y is not None and X.shape[0] != y.shape[-1]:
            raise ValueError("Expected the number of data inputs equal to the number of data "
                             "outputs, but got {} and {}.".format(X.shape[0], y.shape[-1]))
        self.X = X
        self.y = y

    def model(self):
        """
        A "model" stochastic method.
        """
        raise NotImplementedError

    def guide(self):
        """
        A "guide" stochastic method.
        """
        raise NotImplementedError

    def optimize(self, optimizer=Adam({}), num_steps=1000):
        """
        A convenient method to optimize parameters for the Gaussian Process model
        using SVI.

        :param pyro.optim.PyroOptim optimizer: Optimizer.
        :param int num_steps: Number of steps to run SVI.
        :returns: losses of the training procedure
        :rtype: list
        """
        if not isinstance(optimizer, PyroOptim):
            raise ValueError("Optimizer should be an instance of pyro.optim.PyroOptim class.")
        svi = SVI(self.model, self.guide, optimizer, loss="ELBO")
        losses = []
        for i in range(num_steps):
            losses.append(svi.step())
        return losses

    def forward(self, *args, **kwargs):
        """
        Implements prediction step.
        """
        raise NotImplementedError

    def _check_Xnew_shape(self, Xnew):
        """
        Checks the correction of the shape of new data.
        """
        if Xnew.dim() != self.X.dim():
            raise ValueError("Train data and test data should have the same number of dimensions.")
        if Xnew.dim() == 2 and self.X.shape[1] != Xnew.shape[1]:
            raise ValueError("Train data and test data should have the same feature sizes.")

    def _conditional(self, Xnew, X, kernel, f_loc, f_scale_tril=None, Lff=None, full_cov=False):
        """
        Computes the parameters of :math:`p(f^*|Xnew) \sim N(\\text{loc}, \\text{cov})`
        according to :math:`p(f^*,f|y) = p(f^*|f)p(f|y) \sim p(f^*|f)q(f)`,
        then marginalize out variable :math:`f`.
        Here :math:`q(f)` is parameterized by :math:`q(f) \sim N(mf, Lf)`.
        """
        # Ref: https://www.prowler.io/sparse-gps-approximate-the-posterior-not-the-model/
        # p(f* | Xnew, X, kernel, f_loc, f_scale_tril) ~ N(f* | loc, cov)
        # Kff = Lff @ Lff.T
        # v = inv(Lff) @ f_loc  <- whitened f_loc
        # S = inv(Lff) @ f_scale_tril  <- whitened f_scale_tril
        # Denote:
        #     W = inv(Lff) @ Kf*
        #     K = W.T @ S @ S.T @ W
        #     Q** = K*f @ inv(Kff) @ Kf* = W.T @ W
        # loc = K*f @ inv(Kff) @ f_loc = W.T @ v
        # Case 1: f_scale_tril = None
        #     cov = K** - K*f @ inv(Kff) @ Kf* = K** - Q**
        # Case 2: f_scale_tril != None
        #     cov = K** - Q** + K*f @ inv(Kff) @ f_cov @ inv(Kff) @ Kf*
        #         = K** - Q** + W.T @ S @ S.T @ W
        #         = K** - Q** + K

        N = X.shape[0]
        M = Xnew.shape[0]
        latent_shape = f_loc.shape[:-1]

        if Lff is None:
            Kff = kernel(X) + self.jitter.expand(N).diag()
            Lff = Kff.potrf(upper=False)
        Kfs = kernel(X, Xnew)

        # convert f_loc_shape from latent_shape x N to N x latent_shape
        f_loc = f_loc.permute(-1, *range(len(latent_shape)))
        # convert f_loc to 2D tensor for packing
        f_loc_2D = f_loc.reshape(N, -1)
        pack = torch.cat((f_loc_2D, Kfs), dim=1)
        if f_scale_tril is not None:
            # convert f_scale_tril_shape from latent_shape x N x N to N x N x latent_shape
            f_scale_tril = f_scale_tril.permute(-2, -1, *range(len(latent_shape)))
            # convert f_scale_tril to 2D tensor for packing
            f_scale_tril_2D = f_scale_tril.reshape(N, -1)
            pack = torch.cat((pack, f_scale_tril_2D), dim=1)

        Lffinv_pack = matrix_triangular_solve_compat(pack, Lff, upper=False)
        # unpack
        v_2D = Lffinv_pack[:, :f_loc_2D.shape[1]]
        W = Lffinv_pack[:, f_loc_2D.shape[1]:f_loc_2D.shape[1] + M]
        Wt = W.t()

        loc_shape = latent_shape + (M,)
        loc = v_2D.t().matmul(W).reshape(loc_shape)

        if full_cov:
            Kss = kernel(Xnew)
            Qss = Wt.matmul(W)
            cov = Kss - Qss
        else:
            Kssdiag = kernel(Xnew, diag=True)
            Qssdiag = (W ** 2).sum(dim=0)
            var = Kssdiag - Qssdiag

        if f_scale_tril is not None:
            # unpack
            S_2D = Lffinv_pack[:, -f_scale_tril_2D.shape[1]:]
            Wt_S_shape = (Xnew.shape[0],) + f_scale_tril.shape[1:]
            Wt_S = Wt.matmul(S_2D).reshape(Wt_S_shape)
            # convert Wt_S_shape from M x N x latent_shape to latent_shape x M x N
            Wt_S = Wt_S.permute(list(range(2, Wt_S.dim())) + [0, 1])

            if full_cov:
                St_W = Wt_S.transpose(-2, -1)
                K = Wt_S.matmul(St_W)
                cov = cov + K
            else:
                Kdiag = (Wt_S ** 2).sum(dim=-1)
                var = var + Kdiag
        else:
            if full_cov:
                cov = cov.expand(latent_shape + (M, M))
            else:
                var = var.expand(latent_shape + (M,))

        return (loc, cov) if full_cov else (loc, var)
