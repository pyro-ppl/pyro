# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import funsor
import torch
from funsor.adjoint import AdjointTape
from funsor.constant import Constant

from pyro.contrib.funsor import to_data, to_funsor
from pyro.contrib.funsor.handlers import enum, plate, replay, trace
from pyro.contrib.funsor.infer import config_enumerate
from pyro.distributions.util import copy_docs_from
from pyro.infer import Trace_ELBO as _OrigTrace_ELBO

from .elbo import ELBO, Jit_ELBO
from .traceenum_elbo import apply_optimizer, terms_from_trace


@copy_docs_from(_OrigTrace_ELBO)
class Trace_ELBO(ELBO):
    def differentiable_loss(self, model, guide, *args, **kwargs):
        with enum(), plate(
            size=self.num_particles
        ) if self.num_particles > 1 else contextlib.ExitStack():
            guide_tr = trace(
                config_enumerate(default="flat", num_samples=self.num_particles)(guide)
            ).get_trace(*args, **kwargs)
            model_tr = trace(replay(model, trace=guide_tr)).get_trace(*args, **kwargs)

        model_terms = terms_from_trace(model_tr)
        guide_terms = terms_from_trace(guide_tr)

        # cost terms
        costs = model_terms["log_factors"] + [-f for f in guide_terms["log_factors"]]

        plate_vars = guide_terms["plate_vars"] | model_terms["plate_vars"]
        # compute log_measures corresponding to each cost term
        # the goal is to achieve fine-grained Rao-Blackwellization
        targets = dict()
        for cost in costs:
            if cost.input_vars not in targets:
                targets[cost.input_vars] = Constant(
                    cost.inputs, funsor.Tensor(torch.tensor(0.0))
                )
        with AdjointTape() as tape:
            logzq = funsor.sum_product.sum_product(
                funsor.ops.logaddexp,
                funsor.ops.add,
                guide_terms["log_measures"] + list(targets.values()),
                plates=plate_vars,
                eliminate=(plate_vars | guide_terms["measure_vars"]),
            )
        log_measures = tape.adjoint(
            funsor.ops.logaddexp, funsor.ops.add, logzq, tuple(targets.values())
        )
        with funsor.terms.eager:
            # finally, integrate out guide variables in the elbo and all plates
            elbo = to_funsor(0, output=funsor.Real)
            for cost in costs:
                target = targets[cost.input_vars]
                log_measure = log_measures[target]
                measure_vars = frozenset(cost.inputs) - plate_vars
                elbo_term = funsor.Integrate(
                    log_measure,
                    cost,
                    measure_vars,
                )
                elbo += elbo_term.reduce(
                    funsor.ops.add, plate_vars & frozenset(cost.inputs)
                )

        # evaluate the elbo, using memoize to share tensor computation where possible
        with funsor.interpretations.memoize():
            return -to_data(apply_optimizer(elbo))


class JitTrace_ELBO(Jit_ELBO, Trace_ELBO):
    pass
