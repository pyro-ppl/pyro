from __future__ import absolute_import, division, print_function

import pyro
import pyro.poutine as poutine
from pyro.infer.svi import SVI
from pyro.infer.util import is_validation_enabled, torch_item


def _check_params_not_dymanically_generated(org_params, new_params):
    for p in new_params:
        if p not in org_params:
            raise ValueError("Param `{}` is not available in original traces."
                             .format(pyro.get_param_store().param_name(p)))


class StaticSVI(SVI):
    """
    An interface for stochastic variational inference with params in model and guide
    not dynamically generated during the optimization.

    :param callable model: the model (callable containing Pyro primitives)
    :param callable guide: the guide (callable containing Pyro primitives)
    :param ~pyro.optim.PyroOptim optim: a wrapper for a PyTorch optimizer
    :param ~pyro.infer.elbo.ELBO loss: an instance of a subclass of :class:`~pyro.infer.elbo.ELBO`.
    :param callable loss_and_grads: a function which takes inputs are `model`, `guide`,
        and their arguments, computes loss, runs backward, and returns the loss
    :param int num_samples: the number of samples for Monte Carlo posterior approximation
    :param int num_steps: the number of optimization steps to take in ``run()``
    """
    def __init__(self, model, guide, optim, loss, loss_and_grads=None,
                 num_samples=10, num_steps=0, **kwargs):
        super(StaticSVI, self).__init__(model, guide, optim, loss, loss_and_grads,
                                        num_samples, num_steps, **kwargs)
        self._params = None

    def _check_optim(self, optim):
        if not isinstance(optim, pyro.optim.PyroOptim):
            raise ValueError("Optimizer should be an instance of pyro.optim.PyroOptim class.")

    def _setup(self, *args, **kwargs):
        with poutine.trace(param_only=True) as param_capture:
            self.loss_and_grads(self.model, self.guide, *args, **kwargs)

        self._params = set(site["value"].unconstrained()
                           for site in param_capture.trace.nodes.values())

        self._pt_optim = self.optim.pt_optim_constructor(self._params, **self.optim.pt_optim_args)

    def step(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the model and guide
        """
        if self._params is None:
            self._setup(*args, **kwargs)

        def closure():
            pyro.infer.util.zero_grads(self._params)

            if is_validation_enabled():
                with poutine.trace(param_only=True) as param_capture:
                    loss = self.loss_and_grads(self.model, self.guide, *args, **kwargs)

                params = set(site["value"].unconstrained()
                             for site in param_capture.trace.nodes.values())

                _check_params_not_dymanically_generated(self._params, params)
            else:
                loss = self.loss_and_grads(self.model, self.guide, *args, **kwargs)

            return loss

        loss = self._pt_optim.step(closure)

        return torch_item(loss)
