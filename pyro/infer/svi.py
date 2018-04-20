from __future__ import absolute_import, division, print_function

import torch

import pyro
from pyro.infer.elbo import ELBO


class SVI(object):
    """
    :param model: the model (callable containing Pyro primitives)
    :param guide: the guide (callable containing Pyro primitives)
    :param optim: a wrapper a for a PyTorch optimizer
    :type optim: pyro.optim.PyroOptim
    :param loss: an instance of a subclass of :class:`~pyro.infer.elbo.ELBO`.
        Pyro provides three built-in losses:
        :class:`~pyro.infer.trace_elbo.Trace_ELBO`,
        :class:`~pyro.infer.tracegraph_elbo.Trace_ELBO`, and
        :class:`~pyro.infer.traceenum_elbo.Trace_ELBO`.
        See the :class:`~pyro.infer.elbo.ELBO` docs to learn how to implement
        a custom loss.
    :type loss: pyro.infer.elbo.ELBO

    A unified interface for stochastic variational inference in Pyro. The most
    commonly used loss is ``loss=Trace_ELBO()``. See the tutorial
    `SVI Part I <http://pyro.ai/examples/svi_part_i.html>`_ for a discussion.
    """
    def __init__(self,
                 model,
                 guide,
                 optim,
                 loss,
                 loss_and_grads=None,
                 **kwargs):
        self.model = model
        self.guide = guide
        self.optim = optim

        if isinstance(loss, ELBO):
            self.ELBO = loss
            self.loss = self.ELBO.loss
            self.loss_and_grads = self.ELBO.loss_and_grads
        else:
            raise TypeError("Unsupported loss type. Expected an ELBO instance, got a {}".format(type(loss)))

    def __call__(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Convenience method for doing a gradient step.
        """
        self.step(*args, **kwargs)

    def evaluate_loss(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Evaluate the loss function. Any args or kwargs are passed to the model and guide.
        """
        with torch.no_grad():
            return self.loss(self.model, self.guide, *args, **kwargs)

    def step(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype: float

        Take a gradient step on the loss function (and any auxiliary loss functions
        generated under the hood by `loss_and_grads`).
        Any args or kwargs are passed to the model and guide
        """
        # get loss and compute gradients
        loss = self.loss_and_grads(self.model, self.guide, *args, **kwargs)

        # get active params
        params = pyro.get_param_store().get_active_params()

        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        self.optim(params)

        # zero gradients
        pyro.infer.util.zero_grads(params)

        # mark parameters in the param store as inactive
        pyro.get_param_store().mark_params_inactive(params)

        return loss
