from __future__ import absolute_import, division, print_function

import copy

import pyro
from pyro.infer import ELBO


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
                 *args,
                 **kwargs):
        self.model = model
        self.guide = guide
        self.optim = optim

        if isinstance(loss, str):
            assert loss in ["ELBO"], "The only built-in loss currently supported by SVI is ELBO"

            if loss == "ELBO":
                self.ELBO = ELBO(*args, **kwargs)
                self.loss = self.ELBO.loss
                self.loss_and_grads = self.ELBO.loss_and_grads
            else:
                raise NotImplementedError

        else:  # the user provided a loss function
            self.loss = loss
            if loss_and_grads is None:
                # default implementation of loss_and_grads:
                # marks all parameters in param store as active
                # and calls backward() on loss
                # TODO: clean this up

                self._loss = copy.copy(loss)

                def new_loss(model, guide, *args, **kwargs):
                    return self._loss(model, guide, *args, **kwargs).data[0]

                self.loss = new_loss

                def loss_and_grads(model, guide, *args, **kwargs):
                    _loss = self._loss(model, guide, *args, **kwargs)
                    _loss.backward()
                    pyro.get_param_store().mark_params_active(pyro.get_param_store().get_all_param_names())
                    return _loss

            self.loss_and_grads = loss_and_grads

    def __call__(self, *args, **kwargs):
        """
        :returns: estimate of the loss
        :rtype:  float

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
        pyro.util.zero_grads(params)

        # mark parameters in the param store as inactive
        pyro.get_param_store().mark_params_inactive(params)

        return loss
