from __future__ import absolute_import, division, print_function

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
        pt_optim_constructor = optim_args.pop('optimizer')
        # kwargs for the torch optimizer
        optim_kwargs = optim_args.pop('optim_args')
        self.kwargs = optim_args
        # current epoch
        self.epoch = None
        super(PyroLRScheduler, self).__init__(pt_optim_constructor, optim_kwargs)

    def __call__(self, params, *args, **kwargs):
        kwargs['epoch'] = self.epoch
        super(PyroLRScheduler, self).__call__(params, *args, **kwargs)

    def _get_optim(self, params):
        optim = super(PyroLRScheduler, self)._get_optim(params)
        return self.pt_scheduler_constructor(optim, **self.kwargs)

    def set_epoch(self, epoch):
        self.epoch = epoch
