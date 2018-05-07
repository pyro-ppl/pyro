from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
from pyro.distributions.util import matrix_triangular_solve_compat
from pyro.params import param_with_module_name


class Parameterized(nn.Module):
    """
    Base class for other modules in Gaussin Process module.

    Parameters of this object can be set priors, set constraints, or fixed to a
    specific value.

    By default, data of a parameter is a float :class:`torch.Tensor` (unless we use
    :func:`torch.set_default_tensor_type` to change default tensor type). To cast these
    parameters to a correct data type or GPU device, we can call methods such as
    :meth:`~torch.nn.Module.double` or :meth:`~torch.nn.Module.cuda`. See
    :class:`torch.nn.Module` for more information.

    :param str name: Name of this object.
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

        :param str param: Name of the parameter.
        :param ~pyro.distributions.distribution.Distribution prior: A Pyro prior
            distribution.
        """
        self._priors[param] = prior

    def set_constraint(self, param, constraint):
        """
        Sets a constraint to a parameter.

        :param str param: Name of the parameter.
        :param ~torch.distributions.constraints.Constraint constraint: A PyTorch
            constraint. See :mod:`torch.distributions.constraints` for a list of
            constraints.
        """
        self._constraints[param] = constraint

    def fix_param(self, param, value=None):
        """
        Fixes a parameter to a specic value. If ``value=None``, fixes the parameter
        to the default value.

        :param str param: Name of the parameter.
        :param torch.Tensor value: Fixed value.
        """
        if value is None:
            value = getattr(self, param).detach()
        self._fixed_params[param] = value

    def set_mode(self, mode, recursive=True):
        """
        Sets ``mode`` of this object to be able to use its parameters in stochastic
        functions. If ``mode="model"``, a parameter with prior will get its value
        from the primitive :func:`pyro.sample`. If ``mode="guide"`` or there is no
        prior on a parameter, :func:`pyro.param` will be called.

        This method automatically sets ``mode`` for submodules which belong to
        :class:`Parameterized` class unless ``recursive=False``.

        :param str mode: Either "model" or "guide".
        :param bool recursive: A flag to tell if we want to set mode for all
            submodules.
        """
        if mode not in ["model", "guide"]:
            raise ValueError("Mode should be either 'model' or 'guide', but got {}."
                             .format(mode))
        if recursive:
            for module in self.children():
                if isinstance(module, Parameterized):
                    module.set_mode(mode)
        for param in self._parameters:
            self._register_param(param, mode)

    def get_param(self, param):
        """
        Gets the current value of a parameter. The correct behavior will depend on
        ``mode`` of this object (see :meth:`set_mode` method).

        :param str param: Name of the parameter.
        """
        if param not in self._registered_params:  # set_mode() has not been called yet
            return getattr(self, param)
        else:
            return self._registered_params[param]

    def _register_param(self, param, mode="model"):
        """
        Registers a parameter to Pyro. It can be seen as a wrapper for
        :func:`pyro.param` and :func:`pyro.sample` primitives.

        :param str param: Name of the parameter.
        :param str mode: Either "model" or "guide".
        """
        if param in self._fixed_params:
            self._registered_params[param] = self._fixed_params[param]
            return
        prior = self._priors.get(param)
        if self.name is None:
            param_name = param
        else:
            param_name = param_with_module_name(self.name, param)

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
            # TODO: consider to init parameter from a prior call instead of mean
            MAP_param = pyro.param(MAP_param_name, prior.mean.detach())
            p = pyro.sample(param_name, dist.Delta(MAP_param))

        self._registered_params[param] = p


def conditional(Xnew, X, kernel, f_loc, f_scale_tril=None, Lff=None, full_cov=False,
                whiten=False, jitter=1e-6):
    """
    Given :math:`X_{new}`, predicts loc and covariance matrix of the conditional
    multivariate normal distribution

    .. math:: p(f^*(X_{new}) \mid X, k, f_{loc}, f_{scale\_tril}).

    Here ``f_loc`` and ``f_scale_tril`` are variation parameters of the variational
    distribution

    .. math:: q(f \mid f_{loc}, f_{scale\_tril}) \sim p(f | X, y),

    where :math:`f` is the function value of the Gaussian Process given input :math:`X`

    .. math:: p(f(X)) \sim \mathcal{N}(0, k(X, X))

    and :math:`y` is computed from :math:`f` by some likelihood function
    :math:`p(y|f)`.

    In case ``f_scale_tril=None``, we consider :math:`f = f_{loc}` and computes

    .. math:: p(f^*(X_{new}) \mid X, k, f).

    In case ``f_scale_tril`` is not ``None``, we follow the derivation from reference
    [1]. For the case ``f_scale_tril=None``, we follow the popular reference [2].

    References:

    [1] `Sparse GPs: approximate the posterior, not the model
    <https://www.prowler.io/sparse-gps-approximate-the-posterior-not-the-model/>`_

    [2] `Gaussian Processes for Machine Learning`,
    Carl E. Rasmussen, Christopher K. I. Williams

    :param torch.Tensor Xnew: A new input data.
    :param torch.Tensor X: An input data to be conditioned on.
    :param ~pyro.contrib.gp.kernels.kernel.Kernel kernel: A Pyro kernel object.
    :param torch.Tensor f_loc: Mean of :math:`q(f)`. In case ``f_scale_tril=None``,
        :math:`f_{loc} = f`.
    :param torch.Tensor f_scale_tril: Lower triangular decomposition of covariance
        matrix of :math:`q(f)`'s .
    :param torch.Tensor Lff: Lower triangular decomposition of :math:`kernel(X, X)`
        (optional).
    :param bool full_cov: A flag to decide if we want to return full covariance
        matrix or just variance.
    :param bool whiten: A flag to tell if ``f_loc`` and ``f_scale_tril`` are
        already transformed by the inverse of ``Lff``.
    :param float jitter: A small positive term which is added into the diagonal part of
        a covariance matrix to help stablize its Cholesky decomposition.
    :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
    :rtype: tuple(torch.Tensor, torch.Tensor)
    """
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
        Kff = kernel(X) + torch.eye(N, out=X.new_empty(N, N)) * jitter
        Lff = Kff.potrf(upper=False)
    Kfs = kernel(X, Xnew)

    # convert f_loc_shape from latent_shape x N to N x latent_shape
    f_loc = f_loc.permute(-1, *range(len(latent_shape)))
    # convert f_loc to 2D tensor for packing
    f_loc_2D = f_loc.reshape(N, -1)
    if f_scale_tril is not None:
        # convert f_scale_tril_shape from latent_shape x N x N to N x N x latent_shape
        f_scale_tril = f_scale_tril.permute(-2, -1, *range(len(latent_shape)))
        # convert f_scale_tril to 2D tensor for packing
        f_scale_tril_2D = f_scale_tril.reshape(N, -1)

    if whiten:
        v_2D = f_loc_2D
        W = matrix_triangular_solve_compat(Kfs, Lff, upper=False)
        if f_scale_tril is not None:
            S_2D = f_scale_tril_2D
    else:
        pack = torch.cat((f_loc_2D, Kfs), dim=1)
        if f_scale_tril is not None:
            pack = torch.cat((pack, f_scale_tril_2D), dim=1)

        Lffinv_pack = matrix_triangular_solve_compat(pack, Lff, upper=False)
        # unpack
        v_2D = Lffinv_pack[:, :f_loc_2D.shape[1]]
        W = Lffinv_pack[:, f_loc_2D.shape[1]:f_loc_2D.shape[1] + M]
        if f_scale_tril is not None:
            S_2D = Lffinv_pack[:, -f_scale_tril_2D.shape[1]:]

    loc_shape = latent_shape + (M,)
    loc = v_2D.t().matmul(W).reshape(loc_shape)

    if full_cov:
        Kss = kernel(Xnew)
        Qss = W.t().matmul(W)
        cov = Kss - Qss
    else:
        Kssdiag = kernel(Xnew, diag=True)
        Qssdiag = W.pow(2).sum(dim=0)
        var = Kssdiag - Qssdiag

    if f_scale_tril is not None:
        Wt_S_shape = (Xnew.shape[0],) + f_scale_tril.shape[1:]
        Wt_S = W.t().matmul(S_2D).reshape(Wt_S_shape)
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
