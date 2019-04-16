from __future__ import absolute_import, division, print_function

import torch

from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam


class TreeCatTrainer(object):
    def __init__(self, model, optim=None):
        M = model.capacity
        E = len(model.edges)
        if optim is None:
            optim = Adam({'lr': 1e-3})
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        self._svi = SVI(model.model, model.guide, optim, elbo)
        self._suffstats = torch.zeros((E, M, M), dtype=torch.float)

    def step(self, data):
        self._svi.step(data, impute=False)
