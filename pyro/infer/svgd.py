# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from abc import ABCMeta, abstractmethod
import math

import torch
from torch.distributions import biject_to

import pyro
from pyro import poutine
from pyro.distributions import Delta
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.infer.autoguide.guides import AutoContinuous
from pyro.infer.autoguide.initialization import init_to_sample
from pyro.distributions.util import copy_docs_from


def vectorize(fn, num_particles, max_plate_nesting):
    def _fn(*args, **kwargs):
        with pyro.plate("num_particles_vectorized", num_particles, dim=-max_plate_nesting - 1):
            return fn(*args, **kwargs)
    return _fn


class _SVGDGuide(AutoContinuous):
    """
    This modification of :class:`AutoContinuous` is used internally in the
    :class:`SVGD` inference algorithm.
    """
    def __init__(self, model):
        super().__init__(model, init_loc_fn=init_to_sample)

    def get_posterior(self, *args, **kwargs):
        svgd_particles = pyro.param("svgd_particles", self._init_loc)
        return Delta(svgd_particles, event_dim=1)


class SteinKernel(object, metaclass=ABCMeta):
    """
    Abstract class for kernels used in the :class:`SVGD` inference algorithm.
    """

    @abstractmethod
    def log_kernel_and_grad(self, particles):
        """
        Compute the component kernels and their gradients.

        :param particles: a tensor with shape (N, D)
        :returns: A pair (`log_kernel`, `kernel_grad`) where `log_kernel` is a (N, N, D)-shaped
            tensor equal to the logarithm of the kernel and `kernel_grad` is a (N, N, D)-shaped
            tensor where the entry (n, m, d) represents the derivative of `log_kernel` w.r.t.
            x_{m,d}, where x_{m,d} is the d^th dimension of particle m.
        """
        raise NotImplementedError


@copy_docs_from(SteinKernel)
class RBFSteinKernel(SteinKernel):
    """
    A RBF kernel for use in the SVGD inference algorithm. The bandwidth of the kernel is chosen from the
    particles using a simple heuristic as in reference [1].

    :param float bandwidth_factor: Optional factor by which to scale the bandwidth, defaults to 1.0.
    :ivar float ~.bandwidth_factor: Property that controls the factor by which to scale the bandwidth
        at each iteration.

    References

    [1] "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm,"
        Qiang Liu, Dilin Wang
    """
    def __init__(self, bandwidth_factor=None):
        """
        :param float bandwidth_factor: Optional factor by which to scale the bandwidth
        """
        self.bandwidth_factor = bandwidth_factor

    def _bandwidth(self, norm_sq):
        """
        Compute the bandwidth along each dimension using the median pairwise squared distance between particles.
        """
        num_particles = norm_sq.size(0)
        index = torch.arange(num_particles)
        norm_sq = norm_sq[index > index.unsqueeze(-1), ...]
        median = norm_sq.median(dim=0)[0]
        if self.bandwidth_factor is not None:
            median = self.bandwidth_factor * median
        assert median.shape == norm_sq.shape[-1:]
        return median / math.log(num_particles + 1)

    @torch.no_grad()
    def log_kernel_and_grad(self, particles):
        delta_x = particles.unsqueeze(0) - particles.unsqueeze(1)  # N N D
        assert delta_x.dim() == 3
        norm_sq = delta_x.pow(2.0)  # N N D
        h = self._bandwidth(norm_sq)  # D
        log_kernel = -(norm_sq / h)  # N N D
        grad_term = 2.0 * delta_x / h  # N N D
        assert log_kernel.shape == grad_term.shape
        return log_kernel, grad_term

    @property
    def bandwidth_factor(self):
        return self._bandwidth_factor

    @bandwidth_factor.setter
    def bandwidth_factor(self, bandwidth_factor):
        """
        :param float bandwidth_factor: Optional factor by which to scale the bandwidth
        """
        if bandwidth_factor is not None:
            assert bandwidth_factor > 0.0, "bandwidth_factor must be positive."
        self._bandwidth_factor = bandwidth_factor


@copy_docs_from(SteinKernel)
class IMQSteinKernel(SteinKernel):
    r"""
    An IMQ (inverse multi-quadratic) kernel for use in the SVGD inference algorithm [1]. The bandwidth of the kernel
    is chosen from the particles using a simple heuristic as in reference [2]. The kernel takes the form

    :math:`K(x, y) = (\alpha + ||x-y||^2/h)^{\beta}`

    where :math:`\alpha` and :math:`\beta` are user-specified parameters and :math:`h` is the bandwidth.

    :param float alpha: Kernel hyperparameter, defaults to 0.5.
    :param float beta: Kernel hyperparameter, defaults to -0.5.
    :param float bandwidth_factor: Optional factor by which to scale the bandwidth, defaults to 1.0.
    :ivar float ~.bandwidth_factor: Property that controls the factor by which to scale the bandwidth
        at each iteration.

    References

    [1] "Stein Points," Wilson Ye Chen, Lester Mackey, Jackson Gorham, Francois-Xavier Briol, Chris. J. Oates.
    [2] "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm," Qiang Liu, Dilin Wang
    """
    def __init__(self, alpha=0.5, beta=-0.5, bandwidth_factor=None):
        """
        :param float alpha: Kernel hyperparameter, defaults to 0.5.
        :param float beta: Kernel hyperparameter, defaults to -0.5.
        :param float bandwidth_factor: Optional factor by which to scale the bandwidth
        """
        assert alpha > 0.0, "alpha must be positive."
        assert beta < 0.0, "beta must be negative."
        self.alpha = alpha
        self.beta = beta
        self.bandwidth_factor = bandwidth_factor

    def _bandwidth(self, norm_sq):
        """
        Compute the bandwidth along each dimension using the median pairwise squared distance between particles.
        """
        num_particles = norm_sq.size(0)
        index = torch.arange(num_particles)
        norm_sq = norm_sq[index > index.unsqueeze(-1), ...]
        median = norm_sq.median(dim=0)[0]
        if self.bandwidth_factor is not None:
            median = self.bandwidth_factor * median
        assert median.shape == norm_sq.shape[-1:]
        return median / math.log(num_particles + 1)

    @torch.no_grad()
    def log_kernel_and_grad(self, particles):
        delta_x = particles.unsqueeze(0) - particles.unsqueeze(1)  # N N D
        assert delta_x.dim() == 3
        norm_sq = delta_x.pow(2.0)  # N N D
        h = self._bandwidth(norm_sq)  # D
        base_term = self.alpha + norm_sq / h
        log_kernel = self.beta * torch.log(base_term)  # N N D
        grad_term = (-2.0 * self.beta) * delta_x / h  # N N D
        grad_term = grad_term / base_term
        assert log_kernel.shape == grad_term.shape
        return log_kernel, grad_term

    @property
    def bandwidth_factor(self):
        return self._bandwidth_factor

    @bandwidth_factor.setter
    def bandwidth_factor(self, bandwidth_factor):
        """
        :param float bandwidth_factor: Optional factor by which to scale the bandwidth
        """
        if bandwidth_factor is not None:
            assert bandwidth_factor > 0.0, "bandwidth_factor must be positive."
        self._bandwidth_factor = bandwidth_factor


class SVGD:
    """
    A basic implementation of Stein Variational Gradient Descent as described in reference [1].

    :param model: The model (callable containing Pyro primitives). Model must be fully vectorized
        and may only contain continuous latent variables.
    :param kernel: a SVGD compatible kernel like :class:`RBFSteinKernel`.
    :param optim: A wrapper for a PyTorch optimizer.
    :type optim: pyro.optim.PyroOptim
    :param int num_particles: The number of particles used in SVGD.
    :param int max_plate_nesting: The max number of nested :func:`pyro.plate` contexts in the model.
    :param str mode: Whether to use a Kernelized Stein Discrepancy that makes use of `multivariate`
        test functions (as in [1]) or `univariate` test functions (as in [2]). Defaults to `univariate`.

    Example usage:

    .. code-block:: python

        from pyro.infer import SVGD, RBFSteinKernel
        from pyro.optim import Adam

        kernel = RBFSteinKernel()
        adam = Adam({"lr": 0.1})
        svgd = SVGD(model, kernel, adam, num_particles=50, max_plate_nesting=0)

        for step in range(500):
            svgd.step(model_arg1, model_arg2)

        final_particles = svgd.get_named_particles()

    References

    [1] "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm,"
        Qiang Liu, Dilin Wang
    [2] "Kernelized Complete Conditional Stein Discrepancy,"
        Raghav Singhal, Saad Lahlou, Rajesh Ranganath
    """
    def __init__(self, model, kernel, optim, num_particles, max_plate_nesting, mode="univariate"):
        assert callable(model)
        assert isinstance(kernel, SteinKernel), "Must provide a valid SteinKernel"
        assert isinstance(optim, pyro.optim.PyroOptim), "Must provide a valid Pyro optimizer"
        assert num_particles > 1, "Must use at least two particles"
        assert max_plate_nesting >= 0
        assert mode in ['univariate', 'multivariate'], "mode must be one of (univariate, multivariate)"

        self.model = vectorize(model, num_particles, max_plate_nesting)
        self.kernel = kernel
        self.optim = optim
        self.num_particles = num_particles
        self.max_plate_nesting = max_plate_nesting
        self.mode = mode

        self.loss = Trace_ELBO().differentiable_loss
        self.guide = _SVGDGuide(self.model)

    def get_named_particles(self):
        """
        Create a dictionary mapping name to vectorized value, of the form ``{name: tensor}``.
        The leading dimension of each tensor corresponds to particles, i.e. this creates a struct of arrays.
        """
        return {site["name"]: biject_to(site["fn"].support)(unconstrained_value)
                for site, unconstrained_value in self.guide._unpack_latent(pyro.param("svgd_particles"))}

    @torch.no_grad()
    def step(self, *args, **kwargs):
        """
        Computes the SVGD gradient, passing args and kwargs to the model,
        and takes a gradient step.

        :return dict: A dictionary of the form {name: float}, where each float
            is a mean squared gradient. This can be used to monitor the convergence of SVGD.
        """
        # compute gradients of log model joint
        with torch.enable_grad(), poutine.trace(param_only=True) as param_capture:
            loss = self.loss(self.model, self.guide, *args, **kwargs)
            loss.backward()

        # get particles used in the _SVGDGuide and reshape to have num_particles leading dimension
        particles = pyro.param("svgd_particles").unconstrained()
        reshaped_particles = particles.reshape(self.num_particles, -1)
        reshaped_particles_grad = particles.grad.reshape(self.num_particles, -1)

        # compute kernel ingredients
        log_kernel, kernel_grad = self.kernel.log_kernel_and_grad(reshaped_particles)

        if self.mode == "multivariate":
            kernel = log_kernel.sum(-1).exp()
            assert kernel.shape == (self.num_particles, self.num_particles)
            attractive_grad = torch.mm(kernel, reshaped_particles_grad)
            repulsive_grad = torch.einsum("nm,nm...->n...", kernel, kernel_grad)
        elif self.mode == "univariate":
            kernel = log_kernel.exp()
            assert kernel.shape == (self.num_particles, self.num_particles, reshaped_particles.size(-1))
            attractive_grad = torch.einsum("nmd,md->nd", kernel, reshaped_particles_grad)
            repulsive_grad = torch.einsum("nmd,nmd->nd", kernel, kernel_grad)

        # combine the attractive and repulsive terms in the SVGD gradient
        assert attractive_grad.shape == repulsive_grad.shape
        particles.grad = (attractive_grad + repulsive_grad).reshape(particles.shape) / self.num_particles

        # compute per-parameter mean squared gradients
        squared_gradients = {site["name"]: value.mean().item()
                             for site, value in self.guide._unpack_latent(particles.grad.pow(2.0))}

        # torch.optim objects gets instantiated for any params that haven't been seen yet
        params = set(site["value"].unconstrained() for site in param_capture.trace.nodes.values())
        self.optim(params)

        # zero gradients
        pyro.infer.util.zero_grads(params)

        # return per-parameter mean squared gradients to user
        return squared_gradients
