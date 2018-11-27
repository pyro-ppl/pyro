from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.distributions import biject_to, constraints, transform_to
from torch.nn import Parameter

import pyro
import pyro.distributions as dist
from pyro.contrib import autoname
from pyro.infer import Trace_ELBO
from pyro.infer.util import torch_backward, torch_item


class Parameterized(nn.Module):
    """
    Base class for other modules in Gaussin Process module.

    Parameters of this object can be set constraints, set priors. This is achieved
    by moving parameter value to a buffer store and creating "root" parameters which
    are used to generate that parameter's value. For example, if we set a contraint
    to a parameter, an "unconstrained" parameter will be created.

    By default, when we set a prior to a parameter, an auto Delta guide will be created.
    We can use the method :meth:`autoguide` to setup other auto guides.

    To fix a parameter to a specific value, it is enough to turn off its "root"
    parameters' ``requires_grad`` flags.

    Note that by default, data of a parameter is a float :class:`torch.Tensor` (unless we
    use :func:`torch.set_default_tensor_type` to change default tensor type). To cast
    these parameters to a correct data type or GPU device, we can call methods such as
    :meth:`~torch.nn.Module.double` or :meth:`~torch.nn.Module.cuda`. See
    :class:`torch.nn.Module` for more information.
    """
    def __init__(self):
        super(Parameterized, self).__init__()
        self._constraints = OrderedDict()
        self._priors = OrderedDict()
        self._guides = OrderedDict()
        self._mode = None

    def __call__(self, *args, **kwargs):
        with autoname.name_count():
            return super(Parameterized, self).__call__(*args, **kwargs)

    def set_constraint(self, name, constraint):
        """
        Sets a constraint to a parameter.

        :param str name: Name of the parameter.
        :param ~torch.distributions.constraints.Constraint constraint: A PyTorch
            constraint. See :mod:`torch.distributions.constraints` for a list of
            constraints.
        """
        if constraint is constraints.real:
            return

        if name in self._parameters:
            # move param to _buffers
            p = self._parameters.pop(name)
            self.register_buffer(name, p)
        elif name in self._buffers:
            p = self._buffers[name]
        else:
            raise ValueError("There is no parameter with name: {}".format(name))

        p_unconstrained = Parameter(transform_to(constraint).inv(p).detach())
        self.register_parameter("{}_unconstrained".format(name), p_unconstrained)
        self._constraints[name] = constraint

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
        self.autoguide(name, dist.Delta)

    def autoguide(self, name, dist_constructor):
        """
        Sets an autoguide for a parameter with name ``name`` (mimic the behavior of module
        :mod:`pyro.contrib.autoguide`).

        ..note:: `dist_constructor` should be one of :class:`~pyro.distributions.Delta`,
            :class:`~pyro.distributions.Normal`, and
            :class:`~pyro.distributions.MultivariateNormal`. More distribution constructor
            will be supported in the future if needed.

        :param str name: Name of the parameter.
        :param dist_constructor: A `~pyro.distributions.distribution.Distribution` constructor.
        """
        if name not in self._priors:
            raise ValueError("There is no prior for parameter: {}".format(name))

        # TODO: create a new argument `autoguide_args` to store other args for other
        # constructors. For example, in LowRankMVN, we need argument `rank`.
        p = self._buffers[name]
        if dist_constructor is dist.Delta:
            p_map = Parameter(p.detach())
            self.register_parameter("{}_map".format(name), p_map)
            self.set_constraint("{}_map".format(name), self._priors[name].support)
            dist_args = set()
        elif dist_constructor is dist.Normal:
            loc = Parameter(biject_to(self._priors[name].support).inv(p).detach())
            scale = Parameter(loc.new_ones(loc.shape))
            self.register_parameter("{}_loc".format(name), loc)
            self.register_parameter("{}_scale".format(name), scale)
            dist_args = {"loc", "scale"}
        elif dist_constructor is dist.MultivariateNormal:
            loc = Parameter(biject_to(self._priors[name].support).inv(p).detach())
            n = loc.size(-1)
            identity = torch.eye(n, out=loc.new_empty(n, n))
            scale_tril = Parameter(identity.repeat(loc.shape[:-1] + (1, 1)))
            self.register_parameter("{}_loc".format(name), loc)
            self.register_parameter("{}_scale_tril".format(name), scale_tril)
            dist_args = {"loc", "scale_tril"}
        else:
            raise ValueError("Currently, only support autoguide for Delta, Normal, "
                             "and MultivariateNormal distributions.")

        for arg in dist_args:
            self.set_constraint("{}_{}".format(name, arg), dist_constructor.arg_constraints[arg])
        self._guides[name] = (dist_constructor, dist_args)

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
        for module in self.modules():
            if isinstance(module, Parameterized):
                module.mode = mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode
        # We should get buffer values for constrained params first
        # otherwise, autoguide will use the old buffer for `scale` or `scale_tril`
        for name in self._constraints:
            if name not in self._priors:
                self._register_param(name)
        for name in self._priors:
            self._register_param(name)

    def _sample_from_guide(self, name):
        dist_constructor, dist_args = self._guides[name]

        if dist_constructor is dist.Delta:
            p_map = getattr(self, "{}_map".format(name))
            return pyro.sample(name, dist.Delta(p_map).independent())

        # create guide
        dist_args = {arg: getattr(self, "{}_{}".format(name, arg)) for arg in dist_args}
        guide = dist_constructor(**dist_args)

        # no need to do transforms when support is real
        if self._priors[name].support is constraints.real:
            return pyro.sample(name, guide.independent())

        # otherwise, we do inference in unconstrained space and transform the value
        # back to original space
        unconstrained_value = pyro.sample("{}_latent".format(name), guide.independent(),
                                          infer={"is_auxiliary": True})
        transform = biject_to(self._priors[name].support)
        value = transform(unconstrained_value)
        log_density = transform.inv.log_abs_det_jacobian(value, unconstrained_value)
        p = pyro.sample(name, dist.Delta(value, log_density.sum(), event_dim=value.dim()))
        return p

    def _register_param(self, name):
        """
        In "model" mode, lifts the parameter with name ``name`` to a random sample using
        a predefined prior (from :meth:`set_prior` method).

        :param str name: Name of the parameter.
        """
        if name in self._priors:
            with autoname.scope(prefix=self._get_name()):
                if self.mode == "model":
                    p = pyro.sample(name, self._priors[name])
                else:
                    p = self._sample_from_guide(name)
        elif name in self._constraints:
            p_unconstrained = self._parameters["{}_unconstrained".format(name)]
            p = transform_to(self._constraints[name])(p_unconstrained)
        self.register_buffer(name, p)


def train(gpmodule, optimizer=None, loss_fn=None, num_steps=1000):
    """
    A helper to optimize parameters for a GP module.

    :param ~pyro.contrib.gp.models.GPModel gpmodule: A GP module.
    :param ~torch.optim.Optimizer optimizer: A PyTorch optimizer instance.
        By default, we use :class:`~torch.optim.Adam` with ``lr=0.01``.
    :param callable loss_fn: A loss function which takes inputs are
        ``gpmodule.model``, ``gpmodule.guide``, and returns ELBO loss.
        By default, ``loss_fn=Trace_ELBO().differentiable_loss``.
    :param int num_steps: Number of steps to run SVI.
    :returns: a list of losses during the training procedure
    :rtype: list
    """
    optimizer = (torch.optim.Adam(gpmodule.parameters(), lr=0.01)
                 if optimizer is None else optimizer)
    loss_fn = Trace_ELBO().differentiable_loss if loss_fn is None else loss_fn

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(gpmodule.model, gpmodule.guide)
        torch_backward(loss)
        return loss

    losses = []
    for i in range(num_steps):
        loss = optimizer.step(closure)
        losses.append(torch_item(loss))

    return losses


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
