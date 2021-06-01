# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Optional, Union

from torch.optim import lr_scheduler

from pyro.optim.optim import PyroOptim


class PyroLRScheduler(PyroOptim):
    """
    A wrapper for :class:`~torch.optim.lr_scheduler` objects that adjusts learning rates
    for dynamically generated parameters.

    :param scheduler_constructor: a :class:`~torch.optim.lr_scheduler`
    :param optim_args: a dictionary of learning arguments for the optimizer or a callable that returns
        such dictionaries. must contain the key 'optimizer' with pytorch optimizer value
    :param clip_args: a dictionary of clip_norm and/or clip_value args or a callable that returns
        such dictionaries.

    Example::

        optimizer = torch.optim.SGD
        scheduler = pyro.optim.ExponentialLR({'optimizer': optimizer, 'optim_args': {'lr': 0.01}, 'gamma': 0.1})
        svi = SVI(model, guide, scheduler, loss=TraceGraph_ELBO())
        for i in range(epochs):
            for minibatch in DataLoader(dataset, batch_size):
                svi.step(minibatch)
            scheduler.step()
    """
    def __init__(self, scheduler_constructor: lr_scheduler, optim_args: Union[Dict, Callable[..., Dict]],
                 clip_args: Optional[Union[Dict, Callable[..., Dict]]] = None):
        # pytorch scheduler
        self.pt_scheduler_constructor = scheduler_constructor
        # torch optimizer
        pt_optim_constructor = optim_args.pop('optimizer')
        # kwargs for the torch optimizer
        optim_kwargs = optim_args.pop('optim_args')
        self.kwargs = optim_args
        super().__init__(pt_optim_constructor, optim_kwargs, clip_args)

    def __call__(self, params: Dict, *args, **kwargs) -> None:
        super().__call__(params, *args, **kwargs)

    def _get_optim(self, params: Dict):
        optim = super()._get_optim(params)
        return self.pt_scheduler_constructor(optim, **self.kwargs)

    def step(self, *args, **kwargs) -> None:
        """
        Takes the same arguments as the PyTorch scheduler
        (e.g. optional ``loss`` for ``ReduceLROnPlateau``)
        """
        for scheduler in self.optim_objs.values():
            scheduler.step(*args, **kwargs)
