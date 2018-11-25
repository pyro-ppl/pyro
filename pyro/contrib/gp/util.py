from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.distributions import biject_to, constraints

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
    def __init__(self):
        super(Parameterized, self).__init__()
        self._constraints = {}
        self._priors = {}
        self._guides = {}

    @autoname.name_count
    def __call__(self, *args, **kwargs):
        super(Parameterized, self).__call__(*args, **kwargs)

    def set_constraint(self, name, constraint):
        """
        Sets a constraint to a parameter.

        :param str name: Name of the parameter.
        :param ~torch.distributions.constraints.Constraint constraint: A PyTorch
            constraint. See :mod:`torch.distributions.constraints` for a list of
            constraints.
        """
        if name in self._parameters:
            # move param to _buffers
            p = self._parameters.pop(name)
            self.register_buffer(name, p)
        elif name in self._buffers:
            p = self._buffers[name]
        else:
            raise ValueError("There is no parameter with name: {}".format(name))

        p_unconstrained = nn.Parameter(transform_to(constraint).inv(p).detach())
        self.register_parameter("{}_unconstrained".format(name), p_unconstrained)
        self._constraints[param] = constraint

    def set_prior(self, name, prior):
        """
        Sets a prior to a parameter.

        :param str name: Name of the parameter.
        :param ~pyro.distributions.distribution.Distribution prior: A Pyro prior
            distribution.
        """
        if name in self._parameters:
            # move param to _buffers
            p = self._parameters.pop(name)
            self.register_buffer(name, p)
        elif name not in self._buffers:
            raise ValueError("There is no parameter with name: {}".format(name))

        self._priors[name] = prior

    def autoguide(self, name, dist_constructor):
        """
        Sets an autoguide for `param` (mimic the behavior of
        :class:`~pyro.contrib.autoguide.AutoGuide`).

        ..note:: `dist_constructor` should be one of :class:`~pyro.distributions.Delta`,
            :class:`~pyro.distributions.Normal`, and
            :class:`~pyro.distributions.MultivariateNormal`. More distribution constructor
            will be supported in the future if needed.

        :param str param: Name of the parameter.
        :param dist_constructor: A `~pyro.distributions.distribution.Distribution` constructor.
        """
        if name not in self._priors:
            raise ValueError("There is no prior for parameter: {}".format(name))

        # TODO: create a new argument `autoguide_args` to store other args for other
        # constructors. For example, in LowRankMVN, we need argument `rank`.
        p = self._buffers[name]
        if dist_constructor is dist.Delta:
            p_map = nn.Paramter(biject_to(self._priors[param].support).inv(p))
            self.register_parameter("{}_map".format(name), p_map)
        elif dist_constructor is dist.Normal:
            loc = nn.Parameter(biject_to(self._priors[param].support).inv(p))
            scale = nn.Parameter(loc.new_ones(loc.shape))
            self.register_parameter("{}_loc".format(name), loc)
            self.register_parameter("{}_scale".format(name), scale)
            self.set_constraint("{}_scale".format(name), constraints.positive)
        elif dist_constructor is dist.MultivariateNormal:
            loc = nn.Parameter(biject_to(self._priors[name].support).inv(p))
            n = loc.size(-1)
            identity = torch.eye(n, out=loc.new_empty(n, n))
            scale_tril = nn.Parameter(identity.repeat(loc.shape[:-1] + (1, 1)))
            self.register_parameter("{}_loc".format(name), loc)
            self.register_parameter("{}_scale_tril".format(name), scale_tril)
            self.set_constraint("{}_scale_tril".format(name), constraints.lower_cholesky)
        else:
            raise ValueError("Currently, only support autoguide for Delta, Normal, "
                             "and MultivariateNormal distributions.")
        self._guides[param] = dist_constructor

    def set_mode(self, mode):
        """
        Sets ``mode`` of this object to be able to use its parameters in stochastic
        functions. If ``mode="model"``, a parameter with prior will get its value
        from the primitive :func:`pyro.sample`. If ``mode="guide"`` or there is no
        prior on a parameter, :func:`pyro.param` will be called.

        This method automatically sets ``mode`` for submodules which belong to
        :class:`Parameterized` class.

        :param str mode: Either "model" or "guide".
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
        # no need to register params if the new mode is the same to current mode
        if self._mode != mode:
            self._mode = mode
            # We should get buffer values for constrained params first
            # otherwise, autoguide will use the old buffer for `scale` or `scale_tril`
            for name in self._constraints:
                if name not in self._priors:
                    self._register_param(name)
            for name in self._priors:
                self._register_param(name)

    def _sample_from_guide(self, name):
        if self._guides[param] is dist.Delta:
            p_map = getattr(self, "{}_map".format(name))
            guide = dist.Delta(p_map)
        elif self._guides[param] is dist.Normal:
            loc = getattr(self, "{}_loc".format(name))
            scale = getattr(self, "{}_scale".format(name))
            guide = dist.Normal(loc, scale)
        elif self._guides[param] is dist.MultivariateNormal:
            loc = getattr(self, "{}_loc".format(name))
            scale_tril = getattr(self, "{}_scale_tril".format(name))
            guide = dist.MultivariateNormal(loc, scale_tril=scale_tril)

        if self._priors[param].support is constraints.real:
            reinterpreted_batch_ndims = self._buffers[name].dim() - guide.event_dim
            p = pyro.sample(name, guide.independent())
        else:
            unconstrained_value = pyro.sample("{}_latent".format(name), guide.independent(),
                                              infer={"is_auxiliary": True})
            transform = biject_to(self._priors[param].support)
            value = transform(unconstrained_value)
            log_density = transform.inv.log_abs_det_jacobian(value, unconstrained_value)
            p = pyro.sample(name, dist.Delta(value, log_density.sum(), event_dim=value.dim()))
        return p

    def _register_param(self, name):
        """
        In "model" mode, lifts the Parameter `param` to a random sample using
        a predefined prior (from `set_prior(param, prior)` call).

        :param str param: Name of the parameter.
        """
        if name in self._priors:
            with autoname.scope(prefix=self._get_name()):
                if self.mode == "model":
                    p = pyro.sample(param, self._priors[param])
                else:
                    if name not in self._guides:
                        self.autoguide(name, dist.Delta)
                    p = self._sample_from_guide(param)
        elif param in self._constraints:
            unconstrained_param = self._parameters("{}_unconstrained".format(param))
            p = transform_to(self._constraints[param])(unconstrained_param)
        self.register_buffer(name, p)


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
