from __future__ import absolute_import, division, print_function

import pyro
from pyro import poutine
from pyro.infer import infer_discrete


class TreeCatServer(object):
    def __init__(self, model):
        self._model = model

    def impute(self, data=None, num_particles=None):
        if data is None:
            data = [None] * len(self._model.features)
        guide_trace = poutine.trace(self._model.guide).get_trace(data)
        model = poutine.replay(self._model.model, guide_trace)
        if num_particles is not None:
            model = pyro.plate("num_particles_vectorized", num_particles, dim=-2)(model)
        return infer_discrete(model)(data, impute=True)
