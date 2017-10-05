import pyro
from pyro.params import module_from_param_with_module_name, user_param_name
from pyro.infer import ELBO


class Optimize(object):
    """
    A unified interface for optimizing loss functions in pyro.
    """
    def __init__(self,
                 model,
                 guide,
                 optim_constructor,
                 optim_args,
                 loss,
                 loss_and_grads=None,
                 auxiliary_optim_constructor=None,
                 auxiliary_optim_args=None,
                 *args,
                 **kwargs):
	"""
	:param model: the model (callable)
	:param guide: the guide (callable)
	:param optim_constructor: the constructor used to construct the pytorch optim
	    used to optimize the loss
	:param optim_args: either a dictionary of arguments passed to the pytorch optim or
	    a callable that returns such dictionaries. in the latter case, the arguments of
	    the callable are parameter names. this allows the user to, e.g., customize learning
	    rates on a per-parameter basis
	:param loss: this is either a string that specifies the loss function to be used (currently
	    the only supported loss is 'ELBO') or a user-provided loss function.
	:param loss_and_grads: <to be filled in>
	:param auxiliary_optim_constructor: like optim_constructor above, but to be used for the
	    auxiliary loss, if relevant
	:param auxiliary_optim_args: like optim_args above, but for the auxiliary optim
	"""
        self.optim_constructor = optim_constructor
        self.auxiliary_optim_constructor = optim_constructor if auxiliary_optim_constructor is None else\
            auxiliary_optim_constructor

        assert callable(optim_args) or isinstance(
            optim_args, dict), "optim_args must be callable that returns defaults or a defaults dictionary"
        if auxiliary_optim_args is not None:
            assert callable(auxiliary_optim_args) or isinstance(
                auxiliary_optim_args, dict), \
                    "auxiliary_optim_args must be a callable that returns defaults or a defaults dictionary"

        self.optim_args = optim_args
        self.auxiliary_optim_args = auxiliary_optim_args
        self.optim_objects = {}

        if isinstance(loss, str):
            assert loss in ["ELBO"], "The only built-in loss supported by Optimize is ELBO"

            if loss=="ELBO":
                self.ELBO = ELBO(model, guide, *args, **kwargs)
                self.loss = self.ELBO.loss
                self.loss_and_grads = self.ELBO.loss_and_grads
            else:
                raise NotImplementedError
        else:
            self.loss = loss
            if loss_and_grads is None:
                raise NotImplementedError("User must specify loss_and_grads")
            else:
                self.loss_and_grads = loss_and_grads

    # XXX decide what __call__ behavior is
    def __call__(self, *args, **kwargs):
        self.step(*args, **kwargs)

    def evaluate_loss(self, *args, **kwargs):
        """
        evaluate the loss function
        """
        self.loss(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        take a gradient step on the loss function
        """

        loss_scope = kwargs.pop('loss_scope', 'default')
        auxiliary_scope = kwargs.pop('auxiliary_scope', 'baseline')

        # get loss and compute gradients
        loss = self.loss_and_grads(*args, **kwargs)

        # loop over relevant params
        loss_params = pyro.get_param_store().get_active_params(scope=loss_scope)
        auxiliary_params = pyro.get_param_store().get_active_params(scope=auxiliary_scope)

        def step_params(params, optim_constructor, optim_args):
            for param in params:

                # if we have not seen this param before, we instantiate an optim for it
                if param not in self.optim_objects:

                    # get our constructor arguments
                    optim_args_dict = self._get_optim_args(param, optim_args)

                    # create a single optim object for that param
                    self.optim_objects[param] = optim_constructor([param], **optim_args_dict)

                # actually perform the step for this optim object
                self.optim_objects[param].step()

            # zero gradients and mark parameters as inactive
            pyro.util.zero_grads(params)
            pyro.get_param_store().mark_params_inactive(params)

        # do gradient steps corresponding to loss and (if present) auxiliary loss
        step_params(loss_params, self.optim_constructor, self.optim_args)
        step_params(auxiliary_params, self.auxiliary_optim_constructor, self.auxiliary_optim_args)

        return loss

    # helper to fetch the optim args if callable
    def _get_optim_args(self, param, optim_args):
        # if we were passed a fct, we call fct with param info
        # arguments are (module name, param name) e.g. ('mymodule', 'bias')
        if callable(optim_args):

            # get param name
            param_name = pyro._param_store.param_name(param)
            mod_name = module_from_param_with_module_name(param_name)
            stripped_param_name = user_param_name(param_name)
            opt_dict = optim_args(mod_name, stripped_param_name)

            # must be dictionary
            assert isinstance(opt_dict, dict), "per-param optim arg must return defaults dictionary"
            return opt_dict
        else:
            return optim_args
