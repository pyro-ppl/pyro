from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat


class Parameterized(nn.Module):
    """
    Parameterized class.

    This is a base class for other classes in Gaussian Process.
    By default, a parameter will be a :class:`torch.nn.Parameter` containing :class:`torch.FloatTensor`.
    To cast them to the correct data type or GPU device, we can call methods such as
    ``.double()``, ``.cuda(device=0)``,...
    See :class:`torch.nn.Module` for more information.

    :param str name: Name of this module.
    """

    def __init__(self, name=None):
        super(Parameterized, self).__init__()
        self._priors = {}
        self._constraints = {}
        self._fixed_params = {}
        self._registered_params = {}

        self.name = name

    def set_prior(self, param, prior):
        """
        Sets a prior to a parameter.

        :param str param: Name of a parameter.
        :param pyro.distributions.distribution.Distribution prior: A prior
            distribution for random variable ``param``.
        """
        self._priors[param] = prior

    def set_constraint(self, param, constraint):
        """
        Sets a constraint to a parameter.

        :param str param: Name of a parameter.
        :param torch.distributions.constraints.Constraint constraint: A Pytorch constraint.
            See :mod:`torch.distributions.constraints` for a list of constraints.
        """
        self._constraints[param] = constraint

    def fix_param(self, param, value=None):
        """
        Fixes a parameter to a specic value. If ``value=None``, fixes the parameter to the
        default value.

        :param str param: Name of a parameter.
        :param torch.Tensor value: A tensor to be fixed to ``param``.
        """
        if value is None:
            value = getattr(self, param).detach()
        self._fixed_params[param] = value

    def set_mode(self, mode):
        """
        Sets ``mode`` for the module to be able to use its parameters in stochastic functions.
        It also sets ``mode`` for submodules which belong to :class:`Parameterized` class.

        :param str mode: Either "model" or "guide".
        """
        if mode not in ["model", "guide"]:
            raise ValueError("Mode should be either 'model' or 'guide', but got {}.".format(mode))
        for module in self.children():
            if isinstance(module, Parameterized):
                module.set_mode(mode)
        for param in self._parameters:
            self._register_param(param, mode)

    def get_param(self, param):
        """
        Gets variable to be used in stochastic functions. The correct behavior will depend on
        the current ``mode`` of the module.

        :param str param: Name of a parameter.
        """
        if param not in self._registered_params:  # set_mode() has not been called yet
            return getattr(self, param)
        else:
            return self._registered_params[param]

    def _register_param(self, param, mode="model"):
        """
        Registers a parameter to Pyro. It can be seen as a wrapper for ``pyro.param()`` and
        ``pyro.sample()`` calls.

        :param str param: Name of a parameter.
        :param str mode: Either "model" or "guide".
        """
        if param in self._fixed_params:
            self._registered_params[param] = self._fixed_params[param]
            return
        prior = self._priors.get(param)
        if self.name is None:
            param_name = param
        else:
            param_name = pyro.param_with_module_name(self.name, param)

        if prior is None:
            constraint = self._constraints.get(param)
            default_value = getattr(self, param)
            if constraint is None:
                p = pyro.param(param_name, default_value)
            else:
                p = pyro.param(param_name, default_value, constraint=constraint)
        elif mode == "model":
            p = pyro.sample(param_name, prior)
        else:  # prior != None and mode = "guide"
            MAP_param_name = param_name + "_MAP"
            MAP_param_0 = torch.tensor(prior.mean.data.clone(), requires_grad=True)
            MAP_param = pyro.param(MAP_param_name, MAP_param_0)
            p = pyro.sample(param_name, dist.Delta(MAP_param))

        self._registered_params[param] = p


def conditional(Xnew, X, kernel, f_loc, f_scale_tril=None,
                Lff=None, full_cov=False, jitter=1e-6):
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
        Kff = kernel(X) + torch.eye(N, out=X.new(N, N)) * jitter
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
        Qssdiag = W.pow(2).sum(dim=0)
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
            Kdiag = Wt_S.pow(2).sum(dim=-1)
            var = var + Kdiag
    else:
        if full_cov:
            cov = cov.expand(latent_shape + (M, M))
        else:
            var = var.expand(latent_shape + (M,))

    return (loc, cov) if full_cov else (loc, var)
