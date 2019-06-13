from __future__ import absolute_import, division, print_function

from pyro.optim.optim import PyroOptim


class PyroLRScheduler(PyroOptim):
    """
    A wrapper for :class:`~torch.optim.lr_scheduler` objects that adjusts learning rates
    for dynamically generated parameters.

    :param scheduler_constructor: a :class:`~torch.optim.lr_scheduler`
    :param optim_args: a dictionary of learning arguments for the optimizer or a callable that returns
        such dictionaries. must contain the key 'optimizer' with pytorch optimizer value

    Example::

        optimizer = torch.optim.SGD
        scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': {'lr': 0.01}, 'gamma': 0.1})
        svi = SVI(model, guide, pyro_scheduler, loss=TraceGraph_ELBO())
        for i in range(epochs):
            for minibatch in DataLoader(dataset, batch_size):
                svi.step(minibatch)
            scheduler.step(epoch=i)
    """
    def __init__(self, scheduler_constructor, optim_args):
        # pytorch scheduler
        self.pt_scheduler_constructor = scheduler_constructor
        # torch optimizer
        pt_optim_constructor = optim_args.pop('optimizer')
        # kwargs for the torch optimizer
        optim_kwargs = optim_args.pop('optim_args')
        self.kwargs = optim_args
        super(PyroLRScheduler, self).__init__(pt_optim_constructor, optim_kwargs)

    def __call__(self, params, *args, **kwargs):
        super(PyroLRScheduler, self).__call__(params, *args, **kwargs)

    def _get_optim(self, params):
        optim = super(PyroLRScheduler, self)._get_optim(params)
        return self.pt_scheduler_constructor(optim, **self.kwargs)

    def step(self, *args, **kwargs):
        """
        Takes the same arguments as the PyTorch scheduler
        (optional ``epoch`` kwarg or ``loss`` in for ``ReduceLROnPlateau``)
        """
        for scheduler in self.optim_objs.values():
            scheduler.step(*args, **kwargs)
