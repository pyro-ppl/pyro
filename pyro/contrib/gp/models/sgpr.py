from __future__ import absolute_import, division, print_function

from torch.autograd import Variable
import torch.nn as nn

import pyro
import pyro.distributions as dist


def _compute_inverse_and_logdet(D, U):
    """
    Calculate inverse and log determinant of matrix (D + U.Ut) where D is a diagonal matrix
        based on "Woodbury matrix identity" and "matrix determinant lemma"
        inv(D + U.Ut) = inv(D) - inv(D).U.inv(I + Ut.inv(D).U).Ut.inv(D)
        log|D + U.Ut| = log|I + Ut.inv(D).U| + log|D|
    """
    Dinv = 1 / D
    # fast way to calculate torch.matmul(U.t(), Dinv.diag())
    Ut_Dinv = U.t() * Dinv
    I = Variable(U.data.new(U.size(1)).fill_(1)).diag()
    K = I + Ut_Dinv.matmul(U)
    
    # Cholesky decomposition and inverse of low dimensional matrix
    L = K.portf(upper=False)
    Linv_Ut_Dinv = L.inverse().matmul(Ut_Dinv)  # TODO: use trtrs
    
    inverse = Dinv.diag() - Linv_Ut_Dinv.t().matmul(Linv_Ut_Dinv)
    logdet = K.det().log() + D.log().sum()
    return inverse, logdet


class SparseGPRegression(nn.Module):
    """
    Gaussian Process regression module.

    :param torch.autograd.Variable X: A tensor of inputs.
    :param torch.autograd.Variable y: A tensor of outputs for training.
    :param pyro.gp.kernels.Kernel kernel: A Pyro kernel object.
    :param pyro.gp.InducingPoints Xu: A tensor of inducing points for spare approximation.
    :param torch.Tensor noise: An optional noise tensor.
    :param str approx: 'DTC', 'FITC', 'VFE'
    :param dict priors: A mapping from kernel parameter's names to priors.
    """
    def __init__(self, X, y, kernel, Xu, noise=None, approx=None, kernel_prior=None, Xu_prior=None):
        super(GPRegression, self).__init__()
        self.X = X
        self.y = y
        self.kernel = kernel
        self.Xu = Xu
        
        self.num_data = self.X.size(0)
        self.num_inducing = self.Xu().size(0)

        # TODO: define noise as a Likelihood (another nn.Module beside kernel),
        # then we can train/set prior to it
        if noise is None:
            self.noise = Variable(X.data.new([1]))
        else:
            self.noise = Variable(noise)
            
        if approx is None:
            self.approx = "VFE"
        elif approx in ["DTC", "FITC", "VFE"]:
            self.approx = approx
        else:
            raise ValueError("Approximation method should be one of DTC, FITC, or VFE.")
        
        self.kernel_prior = kernel_prior if kernel_prior is not None else {}
        self.Xu_prior = Xu_prior if Xu_prior is not None else {}

    def model(self):
        kernel_fn = pyro.random_module(self.kernel.name, self.kernel, self.kernel_priors)
        kernel = kernel_fn()
        
        Xu_fn = pyro.random_module(self.Xu.name, self.Xu, self.Xu_priors)
        Xu = Xu_fn()()  # first call return module, second call return data
        
        Kuf = kernel()
        
        L = Kuu.potrf(upper=False)
        Linv_Kuf = L.inverse().matmul(Kuf)  # TODO: use trtrs
        
        
        K = kernel(self.X) + self.noise.repeat(self.num_data).diag()
        zero_loc = Variable(K.data.new([0]).expand(self.num_data))
        pyro.sample("y", dist.MultivariateNormal(zero_loc, K), obs=self.y)

    def guide(self):
        guide_priors = {}
        for p in self.priors:
            p_MAP_name = pyro.param_with_module_name(self.kernel.name, p) + "_MAP"
            # init params by their prior means
            p_MAP = pyro.param(p_MAP_name, Variable(self.priors[p].analytic_mean().data.clone(),
                                                    requires_grad=True))
            guide_priors[p] = dist.Delta(p_MAP)
        kernel_fn = pyro.random_module(self.kernel.name, self.kernel, guide_priors)
        kernel = kernel_fn()
        return kernel

    def forward(self, Xnew):
        """
        Compute the parameters of `p(y|Xnew) ~ N(loc, covariance_matrix)`
            w.r.t. the new input Xnew.

        :param torch.autograd.Variable Xnew: A 2D tensor.
        :return: loc and covariance matrix of p(y|Xnew).
        :rtype: torch.autograd.Variable and torch.autograd.Variable
        """
        if Xnew.dim() == 2 and self.X.size(1) != Xnew.size(1):
            assert ValueError("Train data and test data should have the same feature sizes.")
        if Xnew.dim() == 1:
            Xnew = Xnew.unsqueeze(1)
        kernel = self.guide()
        K = kernel(self.X) + self.noise.repeat(self.num_data).diag()
        K_xz = kernel(self.X, Xnew)
        K_zx = K_xz.t()
        K_zz = kernel(Xnew)
        # TODO: use torch.trtrs when it supports cuda tensors
        K_inv = K.inverse()
        loc = K_zx.matmul(K_inv.matmul(self.y))
        covariance_matrix = K_zz - K_zx.matmul(K_inv.matmul(K_xz))
        return loc, covariance_matrix
