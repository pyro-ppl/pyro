# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import funsor
from funsor.adjoint import AdjointTape

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
        breakpoint()

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
                    targets[input_vars] = funsor.Tensor(
                        funsor.ops.new_zeros(
                            funsor.tensor.get_default_prototype(),
                            tuple(v.size for v in cost.inputs.values()),
                        ),
                        cost.inputs,
                        cost.dtype,
                    )
            with AdjointTape() as tape:
                logzq = funsor.sum_product.sum_product(
                    funsor.ops.logaddexp,
                    funsor.ops.add,
                    guide_terms["log_measures"] + list(targets.values()),
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
                logzq_local = marginals[target].reduce(
                    funsor.ops.logaddexp, frozenset(cost.inputs) - plate_vars
                )
                log_prob = marginals[target] - logzq_local
                elbo_term = funsor.Integrate(
                    log_prob,
                    cost,
                    guide_terms["measure_vars"] & frozenset(log_prob.inputs),
                )
                elbo += elbo_term.reduce(
                    funsor.ops.add, plate_vars & frozenset(cost.inputs)
                )

            #  elbo = to_funsor(0.0)
            #  for cost in costs:
            #      elbo += cost.reduce(funsor.ops.add, plate_vars & frozenset(cost.inputs))

            #  return -to_data(elbo)
        # evaluate the elbo, using memoize to share tensor computation where possible
        with funsor.interpretations.memoize():
            result = -to_data(apply_optimizer(elbo))
            return result
            # return -to_data(apply_optimizer(elbo))


class JitTrace_ELBO(Jit_ELBO, Trace_ELBO):
    pass
