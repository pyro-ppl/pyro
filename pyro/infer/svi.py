import pyro
from pyro.infer import ELBO


class SVI(object):
    """
    A unified interface for stochastic variational inference in pyro.
    """
    def __init__(self,
                 model,
                 guide,
                 optim,
                 loss,
                 loss_and_grads=None,
                 *args,
                 **kwargs):
        """
        :param model: the model (callable containing pyro primitives)
        :param guide: the guide (callable containing pyro primitives)
        :param loss: this is either a string that specifies the loss function to be used (currently
            the only supported built-in loss is 'ELBO') or a user-provided loss function;
            in the case this is a built-in loss loss_and_grads will be filled accordingly
        :param loss_and_grads: if specified, this user-provided callable computes gradients for use in step()
            and marks which parameters in the param store are to be optimized
        """
        self.optim = optim

        if isinstance(loss, str):
            assert loss in ["ELBO"], "The only built-in loss supported by Optimize is ELBO"

            if loss == "ELBO":
                self.ELBO = ELBO(model, guide, *args, **kwargs)
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
                def loss_and_grads(*args, **kwargs):
                    _loss = self.loss(*args, **kwargs)
                    _loss.backward()
                    pyro.get_param_store().mark_params_active(pyro.get_param_store().get_all_param_names())
                    return _loss
            self.loss_and_grads = loss_and_grads

    def __call__(self, *args, **kwargs):
        self.step(*args, **kwargs)

    def evaluate_loss(self, *args, **kwargs):
        """
        evaluate the loss function
        """
        return self.loss(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        take a gradient step on the loss function
        (and auxiliary loss function if present in loss_and_grads)
        """
        # get loss and compute gradients
        loss = self.loss_and_grads(*args, **kwargs)

        # get active params
        params = pyro.get_param_store().get_active_params()

        # actually perform gradient steps
        # torch.optim objects gets instantiated for any params that haven't been seen yet
        # these optim objects are tracked by the optim factory
        self.optim(params)

        # zero gradients
        pyro.util.zero_grads(params)

        # mark parameters in parma store as inactive
        pyro.get_param_store().mark_params_inactive(params)

        return loss
