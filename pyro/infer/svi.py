from __future__ import absolute_import, division, print_function

import torch

import pyro
import pyro.poutine as poutine
from pyro.infer.abstract_infer import TracePosterior
from pyro.infer.elbo import ELBO
from pyro.infer.util import torch_item


class SVI(TracePosterior):
    """
    A unified interface for stochastic variational inference in Pyro. The most
    commonly used loss is ``loss=Trace_ELBO()``. See the tutorial
    `SVI Part I <http://pyro.ai/examples/svi_part_i.html>`_ for a discussion.

    :param callable model: the model (callable containing Pyro primitives)
    :param callable guide: the guide (callable containing Pyro primitives)
    :param ~pyro.optim.PyroOptim optim: a wrapper for a PyTorch optimizer
    :param ~pyro.infer.elbo.ELBO loss: an instance of a subclass of :class:`~pyro.infer.elbo.ELBO`.
        Pyro provides three built-in losses:
        :class:`~pyro.infer.trace_elbo.Trace_ELBO`,
        :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`, and
        :class:`~pyro.infer.traceenum_elbo.TraceEnum_ELBO`.
        See the :class:`~pyro.infer.elbo.ELBO` docs to learn how to implement
        a custom loss.
    :param callable loss_and_grads: a function which takes inputs are `model`, `guide`,
        and their arguments, computes loss, runs backward, and returns the loss
    :param int num_samples: the number of samples for Monte Carlo posterior approximation
    :param int num_steps: the number of optimization steps to take in ``run()``
    """
    def __init__(self,
                 model,
                 guide,
                 optim,
                 loss,
                 loss_and_grads=None,
                 num_samples=10,
                 num_steps=0,
                 **kwargs):
        self._check_optim(optim)
        self.model = model
        self.guide = guide
        self.optim = optim
        self.num_steps = num_steps
        self.num_samples = num_samples
        super(SVI, self).__init__(**kwargs)

        if isinstance(loss, ELBO):
            self.loss = loss.loss
            self.loss_and_grads = loss.loss_and_grads
        else:
            if loss_and_grads is None:
                def _loss_and_grads(*args, **kwargs):
                    loss_val = loss(*args, **kwargs)
                    loss_val.backward(retain_graph=True)
                    return loss_val
                loss_and_grads = _loss_and_grads
            self.loss = loss
            self.loss_and_grads = loss_and_grads

    def _check_optim(self, optim):
        if not isinstance(optim, pyro.optim.PyroOptim):
            raise ValueError("Optimizer should be an instance of pyro.optim.PyroOptim class.")
        if isinstance(optim.pt_optim_constructor, torch.optim.LBFGS):
            raise ValueError("SVI is not compatible with LBFGS optimizer.")

    def run(self, *args, **kwargs):
        if self.num_steps > 0:
            with poutine.block():
                for i in range(self.num_steps):
                    self.step(*args, **kwargs)
        return super(SVI, self).run(*args, **kwargs)

    def _traces(self, *args, **kwargs):
        for i in range(self.num_samples):
            guide_trace = poutine.trace(self.guide).get_trace(*args, **kwargs)
            model_trace = poutine.trace(poutine.replay(self.model, trace=guide_trace)).get_trace(*args, **kwargs)
            yield model_trace, 1.0

    def evaluate_loss(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Evaluate the loss function. Any args or kwargs are passed to the model and guide.
        """
        with torch.no_grad():
            return torch_item(self.loss(self.model, self.guide, *args, **kwargs))

    def step(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the model and guide
        """
        # get loss and compute gradients
        with poutine.trace(param_only=True) as param_capture:
            loss = self.loss_and_grads(self.model, self.guide, *args, **kwargs)

        params = set(site["value"].unconstrained()
                     for site in param_capture.trace.nodes.values())

        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params)

        # zero gradients
        pyro.infer.util.zero_grads(params)

        return torch_item(loss)
