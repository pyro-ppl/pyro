from __future__ import absolute_import, division, print_function

from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam


class TreeCatTrainer(object):
    def __init__(self, model, optim=None):
        if optim is None:
            optim = Adam({'lr': 1e-3})
        elbo = TraceEnum_ELBO(max_plate_nesting=1)
        self._global_svi = SVI(model.model, model.guide, optim, elbo)
        self._z_cache = []

    def step(self, data):
        loss = self._global_svi.step(self.data, impute=False)
        # TODO define edge_logits and sample
        return loss
