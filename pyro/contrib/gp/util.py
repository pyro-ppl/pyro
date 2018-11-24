from __future__ import absolute_import, division, print_function

from collections import defaultdict

import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.contrib import autoname


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
        self._constraints = defaultdict(constraints.real)
        self._priors = {}
        self._guides = defaultdict(dist.Delta)
        self._registered_params = {}

    @autoname.name_count
    def __call__(self, *args, **kwargs):
        super(Parameterized, self).__call__(*args, **kwargs)

    def set_constraint(self, param, constraint):
        """
        Sets a constraint to a parameter.

        :param str param: Name of the parameter.
        :param ~torch.distributions.constraints.Constraint constraint: A PyTorch
            constraint. See :mod:`torch.distributions.constraints` for a list of
            constraints.
        """
        self._constraints[param] = constraint

    def set_prior(self, param, prior):
        """
        Sets a prior to a parameter.

        :param str param: Name of the parameter.
        :param ~pyro.distributions.distribution.Distribution prior: A Pyro prior
            distribution.
        """
        self._priors[param] = prior

    def set_guide(self, param, guide):
        """
        Sets a prior to a parameter.

        :param str param: Name of the parameter.
        :param str guide: One of "Delta", "Normal", "MultivariateNormal".
        """
        self._guides[param] = guide

    def set_mode(self, mode):
        """
        Sets ``mode`` of this object to be able to use its parameters in stochastic
        functions. If ``mode="model"``, a parameter with prior will get its value
        from the primitive :func:`pyro.sample`. If ``mode="guide"`` or there is no
        prior on a parameter, :func:`pyro.param` will be called.

        This method automatically sets ``mode`` for submodules which belong to
        :class:`Parameterized` class.

        :param str mode: Either "prior" or "guide".
        """
        self.mode = mode
        for module in self.children():
            if isinstance(module, Parameterized):
                module.set_mode(mode)

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        if self._mode != mode:
            self._mode = mode
            for param in self._parameters:
                self._register_param(param)

    def __setattr__(self, name, value):
        if name in self._registered_params:
            del self._registered_params[name]
        super(Parameterized, self).__setattr__(name, value)

    def __getattr__(self, name):
        if name in self._registered_params:
            return self._registered_params[name]
        super(Parameterized, self).__getattr__(name)

    def _register_param(self, param):
        """
        Registers a parameter to Pyro. It can be seen as a wrapper for
        :func:`pyro.param` and :func:`pyro.sample` primitives.

        :param str param: Name of the parameter.
        """
        value = self._parameters[param]
        with autoname.scope(prefix=self.__class__.__name__):
            if param in self._priors:
                if self.mode == "model":
                    p = pyro.sample(param, self._priors[param])
                else:
                    guide = self._guides[param] if param in self._guides else dist.Delta
                    if guide is dist.Delta:
                        p_MAP = pyro.param("{}_MAP".format(param), self._priors[param])
                        p = pyro.sample(param, dist.Delta(p_MAP))
                    elif guide is dist.Normal:
                        loc = pyro.param("{}_loc".format(param),
                                         lambda: value.new_zeros(value.shape))
                        scale = pyro.param("{}_scale".format(param),
                                           lambda: value.new_ones(value.shape),
                                           constraint=constraints.positive)
                        p = pyro.sample(param, dist.Normal(loc, scale).independent(value.dim()))
                    elif guide is dist.MultivariateNormal:
                        n = value.size(-1)
                        loc = pyro.param("{}_loc".format(param),
                                         lambda: value.new_zeros(value.shape))
                        scale_tril = pyro.param("{}_scale_tril".format(param),
                                                lambda: torch.eye(n, out=value.new_empty(n, n))
                                                .repeat(value.shape[:-1] + (1, 1)),
                                                constraint=constraints.lower_cholesky)
                        p = pyro.sample(param,
                                        dist.MultivariateNormal(loc, scale_tril=scale_tril)
                                        .independent(value.dim() - 1))
            else:
                p = pyro.param(param, value, constraint=self._constraints[param])
        self._registered_params[param] = p


def conditional(Xnew, X, kernel, f_loc, f_scale_tril=None, Lff=None, full_cov=False,
                whiten=False, jitter=1e-6):
    r"""
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
    #     W = (inv(Lff) @ Kf*).T
    #     K = W @ S @ S.T @ W.T
    #     Q** = K*f @ inv(Kff) @ Kf* = W @ W.T
    # loc = K*f @ inv(Kff) @ f_loc = W @ v
    # Case 1: f_scale_tril = None
    #     cov = K** - K*f @ inv(Kff) @ Kf* = K** - Q**
    # Case 2: f_scale_tril != None
    #     cov = K** - Q** + K*f @ inv(Kff) @ f_cov @ inv(Kff) @ Kf*
    #         = K** - Q** + W @ S @ S.T @ W.T
    #         = K** - Q** + K

    N = X.shape[0]
    M = Xnew.shape[0]
    latent_shape = f_loc.shape[:-1]

    if Lff is None:
        Kff = kernel(X).contiguous()
        Kff.view(-1)[::N + 1] += jitter  # add jitter to diagonal
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
        W = Kfs.trtrs(Lff, upper=False)[0].t()
        if f_scale_tril is not None:
            S_2D = f_scale_tril_2D
    else:
        pack = torch.cat((f_loc_2D, Kfs), dim=1)
        if f_scale_tril is not None:
            pack = torch.cat((pack, f_scale_tril_2D), dim=1)

        Lffinv_pack = pack.trtrs(Lff, upper=False)[0]
        # unpack
        v_2D = Lffinv_pack[:, :f_loc_2D.shape[1]]
        W = Lffinv_pack[:, f_loc_2D.shape[1]:f_loc_2D.shape[1] + M].t()
        if f_scale_tril is not None:
            S_2D = Lffinv_pack[:, -f_scale_tril_2D.shape[1]:]

    loc_shape = latent_shape + (M,)
    loc = W.matmul(v_2D).t().reshape(loc_shape)

    if full_cov:
        Kss = kernel(Xnew)
        Qss = W.matmul(W.t())
        cov = Kss - Qss
    else:
        Kssdiag = kernel(Xnew, diag=True)
        Qssdiag = W.pow(2).sum(dim=-1)
        var = Kssdiag - Qssdiag

    if f_scale_tril is not None:
        W_S_shape = (Xnew.shape[0],) + f_scale_tril.shape[1:]
        W_S = W.matmul(S_2D).reshape(W_S_shape)
        # convert W_S_shape from M x N x latent_shape to latent_shape x M x N
        W_S = W_S.permute(list(range(2, W_S.dim())) + [0, 1])

        if full_cov:
            St_Wt = W_S.transpose(-2, -1)
            K = W_S.matmul(St_Wt)
            cov = cov + K
        else:
            Kdiag = W_S.pow(2).sum(dim=-1)
            var = var + Kdiag
    else:
        if full_cov:
            cov = cov.expand(latent_shape + (M, M))
        else:
            var = var.expand(latent_shape + (M,))

    return (loc, cov) if full_cov else (loc, var)
