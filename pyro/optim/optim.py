import pyro
from pyro.params import module_from_param_with_module_name, user_param_name
from pyro.infer import ELBO


class Optimize(object):
    def __init__(self, model, guide,
                 optim_constructor, optim_args,
                 loss, loss_and_grads=None,
                 auxiliary_optim_constructor=None, auxiliary_optim_args=None,
                 *args, **kwargs):

        self.optim_constructor = optim_constructor
        self.auxiliary_optim_constructor = auxiliary_optim_constructor

        assert (auxiliary_optim_constructor is not None and auxiliary_optim_args is not None) or \
               (auxiliary_optim_constructor is None and auxiliary_optim_args is None)
        assert callable(optim_args) or isinstance(
            optim_args, dict), "optim_args must be function that returns defaults or a defaults dictionary"
        if auxiliary_optim_args is not None:
            assert callable(auxiliary_optim_args) or isinstance(
                auxiliary_optim_args, dict), \
                    "auxiliary_optim_args must be function that returns defaults or a defaults dictionary"

        self.optim_args = optim_args
        self.auxiliary_optim_args = auxiliary_optim_args
        self.optim_objects = {}

        if isinstance(loss, str):
            assert loss in ["ELBO"], "The only built-in loss supported by Optimize is ELBO"

            if loss=="ELBO":
                self.loss = ELBO(model, guide, *args, **kwargs).loss
                self.loss_and_grads = ELBO(model, guide, *args, **kwargs).loss_and_grads
            else:
                raise NotImplementedError
        else:
            self.loss = loss
            if loss_and_grads is None:
                raise NotImplementedError
            else:
                self.loss_and_grads = loss_and_grads


    # helper to fetch the optim args if callable
    def get_optim_args(self, param, optim_args):
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

    # decide what default call behavior is
    def __call__(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        loss, trainable_params_dict, baseline_loss, baseline_params = self.loss_and_grad(*args, **kwargs)

        assert (baseline_loss is not None and baseline_params is not None) or \
               (baseline_loss is None and baseline_params is None)

        # we collect all relevant optim objects
        active_optims = []

        # loop over relevant params
        for p in trainable_params_dict.values():

            # if we have not seen this param before, we instantiate an optim
            # obj to deal with it
            if p not in self.optim_objects:

                # get our constructor arguments
                default_optim_dict = self.get_optim_args(p, self.optim_args)

                # create a single optim object for that param
                self.optim_objs[p] = self.optim_constructor([p], **default_optim_dict)

            # we add this to the list of params we'll be stepping on
            active_optims.append(self.optim_objects[p])

        # actually perform the step for each optim obj
        for optim in active_optims:
            optim.step(*args, **kwargs)

        # all done, send back loss if calculated
        return loss
