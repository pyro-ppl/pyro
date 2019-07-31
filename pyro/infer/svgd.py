import torch

import pyro
import pyro.poutine as poutine
from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import AutoDelta


def vectorize(fn, num_particles, max_plate_nesting):
    def _fn(*args, **kwargs):
        with pyro.plate("num_particles_vectorized", num_particles, dim=-max_plate_nesting):
            return fn(*args, **kwargs)
    return _fn


class RBFKernel(object):
    def root_bandwidth(self, param):
        return 1.0

    def _log_kernel(self, param):
        n = param.size(0)
        root_h = self.root_bandwidth(param)
        norm_param = param / root_h
        delta_x = norm_param.unsqueeze(0) - norm_param.unsqueeze(1)
        norm_sq = delta_x.reshape(n, n, -1).pow(2.0).sum(-1)
        log_kernel = -norm_sq
        grad_term = (-2.0) * delta / root_h
        return log_kernel, grad_term

    def kernel_and_grads(self, params):
        kernels, grads = {}, {}
        for name, param in params.items():
            kernel, grad = self._log_kernel(param)
            kernels[name] = kernel
            grads[name] = grad
        kernel = torch.exp(sum(kernels.values()))
        grads = self.apply(kernel, grads) #{name: torch.einsum("ab,b...->a...", kernel, grad) for name, grad in grads.items()}
        return kernel, grads

    def apply(self, kernel, grads):
        assert isinstance(grads, dict)
        return {name: torch.einsum("ab,b...->a...", kernel, grad) for name, grad in grads.items()}


class SVGD(object):
    def __init__(self, model, kernel, num_particles, max_plate_nesting):
        self.model = vectorize(model, num_particles, max_plate_nesting)
        self.guide = AutoDelta(model)
        self.kernel = kernel
        self.num_particles = num_particles
        self.max_plate_nesting = max_plate_nesting
        self.loss = Trace_ELBO().differentiable_loss

    def compute_grad(self, *args, **kwargs):
        loss = self.loss(*args, **kwargs)
        loss.backward()

        kernel, grads = self.kernel.kernel_and_grads()

        for name, site in self.prototype_trace.iter_stochastic_nodes():
            print(name)


if __name__ == '__main__':
    def model():
        pyro.sample("z", dist.Normal(torch.zeros(3), 1.0).to_event(1))

    kernel = RBFKernel()
    svgd = SVGD(model, kernel, 10, 0)
    svgd.compute_grad()

