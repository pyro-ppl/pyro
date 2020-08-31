# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import funsor

from pyro.distributions.util import copy_docs_from
from pyro.infer import Trace_ELBO as _OrigTrace_ELBO

from pyro.contrib.funsor import to_data, to_funsor
from pyro.contrib.funsor.handlers import enum, plate, replay, trace
from pyro.contrib.funsor.infer import config_enumerate

from .elbo import Jit_ELBO, ELBO
from .traceenum_elbo import terms_from_trace


@copy_docs_from(_OrigTrace_ELBO)
class Trace_ELBO(ELBO):

    def differentiable_loss(self, model, guide, *args, **kwargs):
        with enum(), \
                plate(size=self.num_particles) if self.num_particles > 1 else contextlib.ExitStack():
            guide_tr = trace(config_enumerate(default="flat")(guide)).get_trace(*args, **kwargs)
            model_tr = trace(replay(model, trace=guide_tr)).get_trace(*args, **kwargs)

        model_terms = terms_from_trace(model_tr)
        guide_terms = terms_from_trace(guide_tr)

        log_measures = guide_terms["log_measures"] + model_terms["log_measures"]
        log_factors = model_terms["log_factors"] + [-f for f in guide_terms["log_factors"]]
        plate_vars = model_terms["plate_vars"] | guide_terms["plate_vars"]
        measure_vars = model_terms["measure_vars"] | guide_terms["measure_vars"]

        elbo = funsor.Integrate(sum(log_measures, to_funsor(0.)),
                                sum(log_factors, to_funsor(0.)),
                                measure_vars)
        elbo = elbo.reduce(funsor.ops.add, plate_vars)

        return -to_data(elbo)


class JitTrace_ELBO(Jit_ELBO, Trace_ELBO):
    pass
