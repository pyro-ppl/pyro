# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import funsor

from pyro.distributions.util import copy_docs_from
from pyro.infer import TraceTMC_ELBO as _OrigTraceTMC_ELBO

from pyro.contrib.funsor import to_data
from pyro.contrib.funsor.handlers import enum, plate, replay, trace

from pyro.contrib.funsor.infer.elbo import ELBO, Jit_ELBO
from pyro.contrib.funsor.infer.traceenum_elbo import terms_from_trace


@copy_docs_from(_OrigTraceTMC_ELBO)
class TraceTMC_ELBO(ELBO):

    def differentiable_loss(self, model, guide, *args, **kwargs):
        with plate(size=self.num_particles) if self.num_particles > 1 else contextlib.ExitStack(), \
                enum(first_available_dim=(-self.max_plate_nesting-1) if self.max_plate_nesting else None):
            guide_tr = trace(guide).get_trace(*args, **kwargs)
            model_tr = trace(replay(model, trace=guide_tr)).get_trace(*args, **kwargs)

        model_terms = terms_from_trace(model_tr)
        guide_terms = terms_from_trace(guide_tr)

        log_measures = guide_terms["log_measures"] + model_terms["log_measures"]
        log_factors = model_terms["log_factors"] + [-f for f in guide_terms["log_factors"]]
        plate_vars = model_terms["plate_vars"] | guide_terms["plate_vars"]
        measure_vars = model_terms["measure_vars"] | guide_terms["measure_vars"]

        with funsor.interpreter.interpretation(funsor.terms.lazy):
            elbo = funsor.sum_product.sum_product(
                funsor.ops.logaddexp, funsor.ops.add,
                log_measures + log_factors,
                eliminate=measure_vars | plate_vars,
                plates=plate_vars
            )

        return -to_data(funsor.optimizer.apply_optimizer(elbo))


class JitTraceTMC_ELBO(Jit_ELBO, TraceTMC_ELBO):
    pass
