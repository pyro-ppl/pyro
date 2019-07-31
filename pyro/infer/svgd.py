import torch

import pyro
import pyro.poutine as poutine
from pyro.infer import Trace_ELBO


def vectorize(fn, num_particles, max_plate_nesting):
    def _fn(*args, **kwargs):
        with pyro.plate("num_particles_vectorized", num_particles, dim=-max_plate_nesting):
            return fn(*args, **kwargs)
    return _fn


class RBFKernel(object):
    """
    A RBF kernel for use in the SVGD inference algorithm. The bandwidth of the kernel is chosen from the data
    using a simple heuristic as in reference [1].

    References:
    [1] "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm,"
        Qiang Liu, Dilin Wang
    """
    def _bandwidth(self, norm_sq):
        """
        Compute the bandwidth along each dimension using the median pairwise distance between particles.
        """
        num_particles = norm_sq.size(0)
        index = torch.arange(num_particles)
        norm_sq = norm_sq[index > index.unsqueeze(-1), ...]
        median = norm_sq.median(dim=0)[0]
        return median / (num_particles + 1).log()

    def _log_kernel_and_grad(self, param):
        num_particles = param.size(0)
        delta_x = param.unsqueeze(0) - param.unsqueeze(1)
        norm_sq = delta_x.reshape(num_particles, num_particles, -1).pow(2.0)
        h = self._bandwidth(norm_sq)
        log_kernel = -norm_sq / h
        grad_term = (-2.0) * delta_x / h.reshape(param.shape[1:])
        return log_kernel, grad_term

    def kernel_and_grads(self, params):
        log_kernels, grads = {}, {}
        for name, param in params.items():
            log_kernel, grad = self._log_kernel_and_grad(param)
            log_kernels[name] = log_kernel
            grads[name] = grad
        kernel = torch.exp(sum(log_kernels.values()))
        grads = self.apply(kernel, grads)
        return kernel, grads

    def apply(self, kernel, grads):
        assert isinstance(grads, dict)
        return {name: torch.einsum("ab,b...->a...", kernel, grad) for name, grad in grads.items()}


class SVGD(object):
    """
    A basic implementation of Stein Variational Gradient Descent as described in reference [1].

    :param model: the model (callable containing Pyro primitives). model must be fully vectorized.
    :param kernel: a SVGD compatible kernel
    :param int num_particles: the number of particles used in SVGD
    :param int max_plate_nesting: the max number of nested :func:`pyro.plate` contexts in the model.

    References:
    [1] "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm,"
        Qiang Liu, Dilin Wang
    """
    def __init__(self, model, kernel, num_particles, max_plate_nesting):
        self.model = vectorize(model, num_particles, max_plate_nesting)
        # TODO: fix circular import hack
        from pyro.contrib.autoguide import AutoDelta
        self.guide = AutoDelta(model)
        self.kernel = kernel
        self.num_particles = num_particles
        self.max_plate_nesting = max_plate_nesting
        self.loss = Trace_ELBO().differentiable_loss

    def compute_grad(self, *args, **kwargs):
        """
        Computes the SVGD gradient, passing *args and **kwargs to the model.
        """
        loss = self.loss(self.model, self.guide, *args, **kwargs)
        loss.backward()

        params = {name: pyro.param('auto_{}'.format(name)) for name, site in self.guide.prototype_trace.iter_stochastic_nodes()}
        for p, v in params.items():
            print(p, v.shape)

        kernel, grads = self.kernel.kernel_and_grads(params)
