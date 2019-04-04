from __future__ import absolute_import, division, print_function

import pyro
from pyro import poutine
from pyro.infer import infer_discrete


class TreeCatServer(object):
    def __init__(self, model):
        self._model = model

    def impute(self, data=None, num_particles=None, temperature=1):
        if data is None:
            data = [None] * len(self._model.features)

        # Sample global parameters from guide.
        guide_trace = poutine.trace(self._model.guide).get_trace(data)
        model = poutine.replay(self._model.model, guide_trace)

        # Sample local latent variables using variable elimination.
        first_available_dim = -2
        if num_particles is not None:
            model = pyro.plate("num_particles_vectorized", num_particles,
                               dim=first_available_dim)(model)
            first_available_dim -= 1
        model = infer_discrete(model, first_available_dim=-1, temperature=temperature)

        # Run the model.
        return model(data, impute=True)
