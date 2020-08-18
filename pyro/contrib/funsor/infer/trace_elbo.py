# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import funsor

from pyro.distributions.util import copy_docs_from
from pyro.infer import Trace_ELBO as OrigTrace_ELBO

from pyro.contrib.funsor import to_data, to_funsor
from pyro.contrib.funsor.handlers import plate, replay, trace

from .elbo import ELBO
from .traceenum_elbo import terms_from_trace


@copy_docs_from(OrigTrace_ELBO)
class Trace_ELBO(ELBO):

    def differentiable_loss(self, model, guide, *args, **kwargs):
        with plate(size=self.num_particles) if self.num_particles > 1 else contextlib.ExitStack():
            guide_tr = trace(guide).get_trace(*args, **kwargs)
            model_tr = trace(replay(model, trace=guide_tr)).get_trace(*args, **kwargs)

        model_terms = terms_from_trace(model_tr)
        guide_terms = terms_from_trace(guide_tr)

        log_measures = guide_terms["log_measures"] + model_terms["log_measures"]
        log_factors = model_terms["log_factors"] + [-f for f in guide_terms["log_factors"]]
        plate_vars = model_terms["plate_vars"] | guide_terms["plate_vars"]
        measure_vars = model_terms["measure_vars"] | guide_terms["measure_vars"]

        elbo = funsor.Integrate(to_funsor(sum(log_measures)), sum(log_factors), measure_vars)
        elbo = elbo.reduce(funsor.ops.add, plate_vars)

        return -to_data(elbo)
