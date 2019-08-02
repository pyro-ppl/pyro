import math
from contextlib import ExitStack  # python 3

import torch
from torch.distributions import biject_to

import pyro
from pyro.distributions import Delta
from pyro.infer.trace_elbo import Trace_ELBO
from pyro.infer.autoguide.guides import AutoDelta
from pyro.infer.autoguide.initialization import init_to_sample
from pyro.distributions.util import sum_rightmost


def vectorize(fn, num_particles, max_plate_nesting):
    def _fn(*args, **kwargs):
        with pyro.plate("num_particles_vectorized", num_particles, dim=-max_plate_nesting - 1):
            return fn(*args, **kwargs)
    return _fn


class _SVGDGuide(AutoDelta):
    """
    This modification of :class:`AutoDelta` is used internally in the
    :class:`SVGD` inference algorithm.
    """
    def __init__(self, model):
        super(_SVGDGuide, self).__init__(model, prefix="svgd", init_loc_fn=init_to_sample)

    def __call__(self, *args, **kwargs):
        """
        An automatic guide with the same ``*args, **kwargs`` as the base ``model``.

        :return: A dict mapping sample site name to sampled value.
        :rtype: dict
        """
        # if we've never run the model before, do so now so we can inspect the model structure
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        plates = self._create_plates()
        result = {}
        for name, site in self.prototype_trace.iter_stochastic_nodes():
            with ExitStack() as stack:
                for frame in site["cond_indep_stack"]:
                    if frame.vectorized:
                        stack.enter_context(plates[frame.name])
                value = pyro.param("{}_{}".format(self.prefix, name),
                                   site["value"].detach(),
                                   constraint=site["fn"].support)
                # The following lines are where this method differs from the AutoDelta version.
                # For SVGD (but not MAP) we need the log det jacobian terms for correctness.
                transform = biject_to(site["fn"].support)
                unconstrained_value = pyro.param("{}_{}".format(self.prefix, name)).unconstrained()
                log_density = transform.inv.log_abs_det_jacobian(value, unconstrained_value)
                log_density = sum_rightmost(log_density, log_density.dim() - value.dim() + site["fn"].event_dim)
                result[name] = pyro.sample(name, Delta(value, log_density=log_density,
                                           event_dim=site["fn"].event_dim))
        return result


class SteinKernel(object):
    pass


class RBFSteinKernel(SteinKernel):
    """
    A RBF kernel for use in the SVGD inference algorithm. The bandwidth of the kernel is chosen from the
    particles using a simple heuristic as in reference [1].

    :param str mode: whether to use a Kernelized Stein Discrepancy that makes use of multivariate
        test functions (as in [1]) or univariate test functions (as in [2]). defaults to 'univariate'

    References

    [1] "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm,"
        Qiang Liu, Dilin Wang
    [2] "Kernelized Complete Conditional Stein Discrepancy,"
        Raghav Singhal, Saad Lahlou, Rajesh Ranganath
    """
    def __init__(self, mode='univariate'):
        assert mode in ['univariate', 'multivariate'], "mode must be one of (univariate, multivariate)"
        self.mode = mode

    def _bandwidth(self, norm_sq):
        """
        Compute the bandwidth along each dimension using the median pairwise squared distance between particles.
        """
        num_particles = norm_sq.size(0)
        index = torch.arange(num_particles)
        norm_sq = norm_sq[index > index.unsqueeze(-1), ...]
        median = norm_sq.median(dim=0)[0]
        return median / math.log(num_particles + 1)

    def _log_kernel_and_grad(self, param, bandwidth_factor=None):
        """
        Compute the component kernels and (parts of) their gradients.

        :param float bandwidth_factor: optional factor by which to scale the bandwidth
        """
        num_particles = param.size(0)
        delta_x = param.unsqueeze(0) - param.unsqueeze(1)
        norm_sq = delta_x.reshape(num_particles, num_particles, -1).pow(2.0)
        h = self._bandwidth(norm_sq)
        if bandwidth_factor:
            h *= bandwidth_factor
        log_kernel = -(norm_sq / h)
        grad_term = (-2.0) * delta_x / h.reshape(param.shape[1:])
        return log_kernel, grad_term

    def kernel_and_grads(self, params, bandwidth_factor=None):
        """
        Compute the full kernel and its gradient

        :param dict params: dictionary of (name, parameter) pairs
        :param float bandwidth_factor: optional factor by which to scale the bandwidth
        """
        log_kernels, grads = {}, {}
        for name, param in params.items():
            log_kernel, grad = self._log_kernel_and_grad(param, bandwidth_factor=bandwidth_factor)
            log_kernels[name] = log_kernel
            grads[name] = grad
        if self.mode == "multivariate":
            kernel = torch.exp(sum([lk.sum(-1) for lk in log_kernels.values()]))
            grads = {name: torch.einsum("ji,ji...->i...", kernel, grad) for name, grad in grads.items()}
        else:
            kernel = {name: lk.exp() for name, lk in log_kernels.items()}
            grads = {name: torch.einsum("j...,j...->...", kernel[name], grad) for name, grad in grads.items()}
        return kernel, grads


class SVGD(object):
    """
    A basic implementation of Stein Variational Gradient Descent as described in reference [1].

    :param model: the model (callable containing Pyro primitives). model must be fully vectorized.
    :param kernel: a SVGD compatible kernel like :class:`RBFSteinKernel`
    :param optim: a wrapper a for a PyTorch optimizer
    :type optim: pyro.optim.PyroOptim
    :param int num_particles: the number of particles used in SVGD
    :param int max_plate_nesting: the max number of nested :func:`pyro.plate` contexts in the model.

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
    def __init__(self, model, kernel, optim, num_particles, max_plate_nesting):
        assert callable(model)
        assert isinstance(kernel, SteinKernel), "Must provide a valid SteinKernel"
        assert isinstance(optim, pyro.optim.PyroOptim), "Must provide a valid Pyro optimizer"
        assert num_particles > 1, "Must use at least two particles"
        assert max_plate_nesting >= 0

        self.model = vectorize(model, num_particles, max_plate_nesting)
        self.kernel = kernel
        self.optim = optim
        self.num_particles = num_particles
        self.max_plate_nesting = max_plate_nesting

        self.loss = Trace_ELBO().differentiable_loss
        self.guide = _SVGDGuide(self.model)

    def get_named_particles(self):
        """
        Get a dictionary of named particles of the form {name: particle}
        """
        params = {name: pyro.param('svgd_{}'.format(name))
                  for name, site in self.guide.prototype_trace.iter_stochastic_nodes()}
        return params

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

        # compute the kernel ingredients needed for SVGD
        params = {name: param.unconstrained() for name, param in self.get_named_particles().items()}
        kernel, kernel_grads = self.kernel.kernel_and_grads(params, bandwidth_factor=bandwidth_factor)

        if self.kernel.mode == "multivariate":
            param_grads = {name: torch.einsum("ij,j...->i...", kernel, param.grad) for name, param in params.items()}
        else:
            param_grads = {name: torch.einsum("ji...,i...->j...", kernel[name], param.grad)
                           for name, param in params.items()}

        # combine the attractive and repulsive terms in the SVGD gradient
        for name, param in params.items():
            assert param_grads[name].shape == kernel_grads[name].shape
            param.grad.data = (param_grads[name] + kernel_grads[name]) / self.num_particles

        if gradient_callback is not None:
            squared_gradients = {name: param.grad.pow(2.0).mean().item() for name, param in params.items()}
            gradient_callback(squared_gradients=squared_gradients)

        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params.values())

        # zero gradients
        pyro.infer.util.zero_grads(params.values())
