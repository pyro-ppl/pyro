from __future__ import absolute_import, division, print_function

import pyro
from pyro.infer.elbo import ELBO


class SVI(object):
    """
    :param model: the model (callable containing Pyro primitives)
    :param guide: the guide (callable containing Pyro primitives)
    :param optim: a wrapper a for a PyTorch optimizer
    :type optim: pyro.optim.PyroOptim
    :param loss: this is either a string that specifies the loss function to be used (currently
        the only supported built-in loss is 'ELBO') or a user-provided loss function;
        in the case this is a built-in loss `loss_and_grads` will be filled in accordingly
    :param loss_and_grads: if specified, this user-provided callable computes gradients for use in `step()`
        and marks which parameters in the param store are to be optimized

    A unified interface for stochastic variational inference in Pyro. Most
    users will interact with `SVI` with the argument `loss="ELBO"`. See the
    tutorial `SVI Part I <http://pyro.ai/examples/svi_part_i.html>`_ for a discussion.
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

        if isinstance(loss, str):
            if loss == "ELBO":
                self.ELBO = ELBO.make(**kwargs)
                self.loss = self.ELBO.loss
                self.loss_and_grads = self.ELBO.loss_and_grads
            else:
                raise NotImplementedError("The only built-in loss currently supported by SVI is ELBO")
        elif isinstance(loss, ELBO):
            self.ELBO = loss
            self.loss = self.ELBO.loss
            self.loss_and_grads = self.ELBO.loss_and_grads
        else:
            raise TypeError("Unsupported loss type {}".format(type(loss)))

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
