import pyro
from pyro.params import module_from_param_with_module_name, user_param_name


class PyroOptim(object):
    """
    A wrapper for torch.optim.Optimizer objects that helps managing with dynamically generated parameters
    """
    def __init__(self, optim_constructor, optim_args):
        """
        :param optim_constructor: a torch.optim.Optimizer
        :param optim_args: a dictionary of learning arguments for the optimizer or a callable that returns
            such dictionaries
        """
        self.pt_optim_constructor = optim_constructor

        # must be callable or dict
        assert callable(optim_args) or isinstance(
            optim_args, dict), "optim_args must be function that returns defaults or a defaults dictionary"

        # hold our args to be called/used
        self.pt_optim_args = optim_args

        # holds the torch optimizer objects
        self.optim_objs = {}

    def __call__(self, params,  *args, **kwargs):
        """
        :param params: a list of parameters
        :type params: an iterable of strings

        Do an optimization step for each param in params. If a given param has never been seen before,
        initialize an optimizer for it.
        """

        for p in params:
            # if we have not seen this param before, we instantiate and optim object to deal with it
            if p not in self.optim_objs:
                # get our constructor arguments
                def_optim_dict = self.get_optim_args(p)
                # create a single optim object for that param
                self.optim_objs[p] = self.pt_optim_constructor([p], **def_optim_dict)

            # actually perform the step for the optim object
            self.optim_objs[p].step(*args, **kwargs)

    # helper to fetch the optim args if callable
    def get_optim_args(self, param):
        # if we were passed a fct, we call fct with param info
        # arguments are (module name, param name, tags) e.g. ('mymodule', 'bias', 'baseline')
        if callable(self.pt_optim_args):

            # get param name
            param_name = pyro._param_store.param_name(param)
            module_name = module_from_param_with_module_name(param_name)
            stripped_param_name = user_param_name(param_name)

            # get tags
            tags = pyro.get_param_store().get_param_tags(param_name)

            # invoke the user-provided callable
            opt_dict = self.pt_optim_args(module_name, stripped_param_name, tags)

            # must be dictionary
            assert isinstance(opt_dict, dict), "per-param optim arg must return defaults dictionary"
            return opt_dict
        else:
            return self.pt_optim_args
