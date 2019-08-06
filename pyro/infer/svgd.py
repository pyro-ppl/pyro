from abc import ABCMeta, abstractmethod
import math
from contextlib import ExitStack  # python 3

import torch
from torch.distributions import biject_to

import pyro
from pyro.distributions import Delta
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.infer.autoguide.guides import AutoContinuous
from pyro.infer.autoguide.initialization import init_to_sample
from pyro.distributions.util import sum_rightmost


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
        super(_SVGDGuide, self).__init__(model, prefix="svgd", init_loc_fn=init_to_sample)

    def get_posterior(self, *args, **kwargs):
        svgd_particles = pyro.param("svgd_particles", self._init_loc)
        return Delta(svgd_particles, event_dim=1)


class SteinKernel(object, metaclass=ABCMeta):

    @abstractmethod
    def log_kernel_and_grad(self, particles):
        """
        Compute the component kernels and (parts of) their gradients.
        """
        raise NotImplementedError


class RBFSteinKernel(SteinKernel):
    """
    A RBF kernel for use in the SVGD inference algorithm. The bandwidth of the kernel is chosen from the
    particles using a simple heuristic as in reference [1].

    References

    [1] "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm,"
        Qiang Liu, Dilin Wang
    """
    def _bandwidth(self, norm_sq, bandwidth_factor=None):
        """
        Compute the bandwidth along each dimension using the median pairwise squared distance between particles.
        """
        num_particles = norm_sq.size(0)
        index = torch.arange(num_particles)
        norm_sq = norm_sq[index > index.unsqueeze(-1), ...]
        median = norm_sq.median(dim=0)[0]
        if bandwidth_factor is not None:
            median = bandwidth_factor * median
        assert median.shape == norm_sq.shape[-1:]
        return median / math.log(num_particles + 1)

    def log_kernel_and_grad(self, particles, bandwidth_factor=None):
        """
        Compute the component kernels and (parts of) their gradients.
        """
        num_particles = particles.size(0)
        delta_x = particles.unsqueeze(0) - particles.unsqueeze(1)  # N N D
        assert delta_x.dim() == 3
        norm_sq = delta_x.pow(2.0)  # N N D
        h = self._bandwidth(norm_sq, bandwidth_factor=bandwidth_factor)  # D
        log_kernel = -(norm_sq / h)  # N N D
        grad_term = 2.0 * delta_x / h  # N N D
        assert log_kernel.shape == grad_term.shape
        return log_kernel, grad_term


class SVGD(object):
    """
    A basic implementation of Stein Variational Gradient Descent as described in reference [1].

    :param model: the model (callable containing Pyro primitives). model must be fully vectorized.
    :param kernel: a SVGD compatible kernel like :class:`RBFSteinKernel`
    :param optim: a wrapper a for a PyTorch optimizer
    :type optim: pyro.optim.PyroOptim
    :param int num_particles: the number of particles used in SVGD
    :param int max_plate_nesting: the max number of nested :func:`pyro.plate` contexts in the model.
    :param str mode: whether to use a Kernelized Stein Discrepancy that makes use of `multivariate`
        test functions (as in [1]) or `univariate` test functions (as in [2]). defaults to `univariate`.

    Example usage:

    .. code-block:: python

        from pyro.infer import SVGD, RBFSteinKernel
        from pyro.optim import Adam

        kernel = RBFSteinKernel()
        adam = Adam({"lr": 0.1})
        svgd = SVGD(model, kernel, adam, num_particles=50, max_plate_nesting=0)

        def gradient_callback(squared_gradients):
            print(squared_gradients)  # this helps us monitor convergence

        for step in range(500):
            svgd.step(model_arg1, model_arg2, gradient_callback=gradient_callback)
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
        Get a dictionary of named particles of the form {name: particle}. The leading dimension
        corresponds to particles.
        """
        return {site["name"]: biject_to(site["fn"].support)(unconstrained_value)
                for site, unconstrained_value in self.guide._unpack_latent(pyro.param("svgd_particles"))}

    def step(self, *args, **kwargs):
        """
        Computes the SVGD gradient, passing args and kwargs to the model,
        and takes a gradient step.

        :param float bandwidth_factor: Optional factor by which to scale the bandwidth
        :param callable gradient_callback: Optional callback that takes a
            single kwarg `squared_gradients`, which is a dictionary of the form
            {param_name: float}, where each float is a mean squared gradient.
            This can be used to monitor the convergence of SVGD.
        """
        bandwidth_factor = kwargs.pop('bandwidth_factor', None)
        gradient_callback = kwargs.pop('gradient_callback', None)
        if gradient_callback is not None:
            assert callable(gradient_callback), "gradient_callback must be a callable"

        # compute gradients of log model joint
        loss = self.loss(self.model, self.guide, *args, **kwargs)
        loss.backward()

        # get particles used in the _SVGDGuide and reshape to have num_particles leading dimension
        particles = pyro.param("svgd_particles").unconstrained()
        reshaped_particles = particles.reshape(self.num_particles, -1)
        reshaped_particles_grad = particles.grad.reshape(self.num_particles, -1)

        # compute kernel ingredients
        log_kernel, kernel_grad = self.kernel.log_kernel_and_grad(reshaped_particles, bandwidth_factor=bandwidth_factor)

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

        # optionally report per-parameter mean squared gradients to user
        if gradient_callback is not None:
            squared_gradients = {site["name"]: value.mean().item()
                                 for site, value in self.guide._unpack_latent(particles.grad.pow(2.0))}
            gradient_callback(squared_gradients=squared_gradients)

        # what about other params???
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim([particles])

        # zero gradients
        pyro.infer.util.zero_grads([particles])
