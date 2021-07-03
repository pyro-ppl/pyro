# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import funsor

from pyro.contrib.funsor import to_data, to_funsor
from pyro.contrib.funsor.handlers import enum, plate, replay, trace
from pyro.contrib.funsor.infer import config_enumerate
from pyro.distributions.util import copy_docs_from
from pyro.infer import Trace_ELBO as _OrigTrace_ELBO

from .elbo import ELBO, Jit_ELBO
from .traceenum_elbo import terms_from_trace


@copy_docs_from(_OrigTrace_ELBO)
class Trace_ELBO(ELBO):
    def differentiable_loss(self, model, guide, *args, **kwargs):
        with enum(), plate(
            size=self.num_particles
        ) if self.num_particles > 1 else contextlib.ExitStack():
            guide_tr = trace(config_enumerate(default="flat")(guide)).get_trace(
                *args, **kwargs
            )
            model_tr = trace(replay(model, trace=guide_tr)).get_trace(*args, **kwargs)

        model_terms = terms_from_trace(model_tr)
        guide_terms = terms_from_trace(guide_tr)

        costs = model_terms["log_factors"] + [-f for f in guide_terms["log_factors"]]
        plate_vars = model_terms["plate_vars"] | guide_terms["plate_vars"]

        elbo = to_funsor(0.0)
        for cost in costs:
            elbo += cost.reduce(funsor.ops.add, plate_vars & frozenset(cost.inputs))

        return -to_data(elbo)


class JitTrace_ELBO(Jit_ELBO, Trace_ELBO):
    pass
