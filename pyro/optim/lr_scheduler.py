from __future__ import absolute_import, division, print_function

import torch

import pyro
from pyro.params import module_from_param_with_module_name, user_param_name


class PyroLRScheduler(object):
    """
    A wrapper for torch.optim.lr_scheduler objects that adjust learning rates
    for dynamically generated parameters.

    :param optim_constructor: a torch.optim.lr_scheduler
    :param optim_args: a dictionary of learning arguments for the optimizer or a callable that returns
        such dictionaries. must contain the key 'optimizer' with pytorch optimizer value

    Example::

        optimizer = torch.optim.SGD
        pyro_scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': {'lr': 0.01}, 'gamma': 0.1})
    """
    def __init__(self, scheduler_constructor, optim_args):
        # pytorch scheduler
        self.pt_scheduler_constructor = scheduler_constructor
        # torch optimizer
        self.pt_optim_constructor = optim_args.pop('optimizer')
        # kwargs for the torch optimizer
        self.kwargs = optim_args.pop('optim_args')

        # must be callable or dict
        assert callable(optim_args) or isinstance(
            optim_args, dict), "optim_args must be function that returns defaults or a defaults dictionary"

        # hold our args to be called/used
        self.pt_optim_args = optim_args

        # holds the torch optimizer objects
        self.optim_objs = {}

        # any optimizer state that's waiting to be consumed (because that parameter hasn't been seen before)
        self._state_waiting_to_be_consumed = {}

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
                def_optim_dict = self._get_optim_args(p)
                # wrap torch lr_scheduler with a pyro lr scheduler
                # create a single optim object for that param
                optim = self.pt_optim_constructor([p], **self.kwargs)
                scheduler = self.pt_scheduler_constructor(optim, **def_optim_dict)
                self.optim_objs[p] = scheduler

                # set state from _state_waiting_to_be_consumed if present
                param_name = pyro.get_param_store().param_name(p)
                if param_name in self._state_waiting_to_be_consumed:
                    state = self._state_waiting_to_be_consumed.pop(param_name)
                    self.optim_objs[p].load_state_dict(state)

            # actually perform the step for the optim object
            self.optim_objs[p].step(*args, **kwargs)

    def get_state(self):
        """
        Get state associated with all the optimizers in the form of a dictionary with
        key-value pairs (parameter name, optim state dicts)
        """
        state_dict = {}
        for param in self.optim_objs:
            param_name = pyro.get_param_store().param_name(param)
            state_dict[param_name] = self.optim_objs[param].state_dict()
        return state_dict

    def set_state(self, state_dict):
        """
        Set the state associated with all the optimizers using the state obtained
        from a previous call to get_state()
        """
        self._state_waiting_to_be_consumed = state_dict

    def save(self, filename):
        """
        :param filename: file name to save to
        :type name: str

        Save optimizer state to disk
        """
        with open(filename, "wb") as output_file:
            torch.save(self.get_state(), output_file)

    def load(self, filename):
        """
        :param filename: file name to load from
        :type name: str

        Load optimizer state from disk
        """
        with open(filename, "rb") as input_file:
            state = torch.load(input_file)
        self.set_state(state)

    # helper to fetch the optim args if callable (only used internally)
    def _get_optim_args(self, param):
        # if we were passed a fct, we call fct with param info
        # arguments are (module name, param name) e.g. ('mymodule', 'bias')
        if callable(self.pt_optim_args):

            # get param name
            param_name = pyro.get_param_store().param_name(param)
            module_name = module_from_param_with_module_name(param_name)
            stripped_param_name = user_param_name(param_name)

            # invoke the user-provided callable
            opt_dict = self.pt_optim_args(module_name, stripped_param_name)

            # must be dictionary
            assert isinstance(opt_dict, dict), "per-param optim arg must return defaults dictionary"
            return opt_dict
        return self.pt_optim_args
