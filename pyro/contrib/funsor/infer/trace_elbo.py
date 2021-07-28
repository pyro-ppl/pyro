# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from collections import OrderedDict

import funsor
import torch
from funsor.adjoint import AdjointTape
from funsor.constant import Constant
from funsor.montecarlo import MonteCarlo

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

        with funsor.terms.eager:
            costs = model_terms["log_factors"] + [
                -f for f in guide_terms["log_factors"]
            ]

            # compute expected cost
            # Cf. pyro.infer.util.Dice.compute_expectation()
            # https://github.com/pyro-ppl/pyro/blob/0.3.0/pyro/infer/util.py#L212
            # TODO Replace this with funsor.Expectation
            plate_vars = guide_terms["plate_vars"] | model_terms["plate_vars"]
            # compute the marginal logq in the guide corresponding to each cost term
            targets = dict()
            for cost in costs:
                input_vars = frozenset(cost.inputs)
                if input_vars not in targets:
                    const_inputs = tuple((v.name, v.output) for v in cost.input_vars)
                    targets[input_vars] = Constant(
                        const_inputs, funsor.Tensor(torch.tensor(0))
                    )
            with AdjointTape() as tape:
                logzq = funsor.sum_product.sum_product(
                    funsor.ops.logaddexp,
                    funsor.ops.add,
                    guide_terms["unsampled_log_measures"] + list(targets.values()),
                    plates=plate_vars,
                    eliminate=(plate_vars | guide_terms["measure_vars"]),
                )
            marginals = tape.adjoint(
                funsor.ops.logaddexp, funsor.ops.add, logzq, tuple(targets.values())
            )
            # finally, integrate out guide variables in the elbo and all plates
            elbo = to_funsor(0, output=funsor.Real)
            for cost in costs:
                target = targets[frozenset(cost.inputs)]
                marginal = marginals[target]
                #  logzq_local = marginals[target].reduce(
                #      funsor.ops.logaddexp, frozenset(cost.inputs) - plate_vars
                #  )
                #  logzq_local = guide_terms["log_measures"][0].reduce(
                #      funsor.ops.logaddexp, frozenset(cost.inputs) - plate_vars
                #  )
                # log_prob = marginal.sample(frozenset(cost.inputs)-plate_vars)  # - logzq + logzq_local
                # log_prob = guide_terms["log_measures"][0]
                measure_vars = frozenset(cost.inputs) - plate_vars
                _raw_value = {var: guide_tr.nodes[var]["value"]._t for var in measure_vars}
                with MonteCarlo(raw_value=_raw_value):
                    elbo_term = funsor.Integrate(
                        marginal,
                        cost,
                        guide_terms["measure_vars"] & frozenset(marginal.inputs),
                    )
                elbo += elbo_term.reduce(
                    funsor.ops.add, plate_vars & frozenset(cost.inputs)
                )

        # evaluate the elbo, using memoize to share tensor computation where possible
        with funsor.interpretations.memoize():
            return -to_data(apply_optimizer(elbo))


class JitTrace_ELBO(Jit_ELBO, Trace_ELBO):
    pass
