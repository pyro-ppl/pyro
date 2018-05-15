from __future__ import absolute_import, division, print_function

import pyro
from pyro.optim.optim import PyroOptim


class PyroLRScheduler(PyroOptim):
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

        self.epoch = None

        # any optimizer state that's waiting to be consumed (because that parameter hasn't been seen before)
        self._state_waiting_to_be_consumed = {}

    def __call__(self, params,  *args, **kwargs):
        """
        :param params: a list of parameters
        :type params: an iterable of strings

        Do an optimization step for each param in params. If a given param has never been seen before,
        initialize an optimizer for it.
        """
        kwargs['epoch'] = self.epoch
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

    def set_epoch(self, epoch):
        self.epoch = epoch
