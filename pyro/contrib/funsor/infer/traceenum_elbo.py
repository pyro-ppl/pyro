# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor

from pyro.infer import ELBO
from pyro.poutine.util import prune_subsample_sites

from pyro.contrib.funsor import to_data, to_funsor
from pyro.contrib.funsor.handlers import enum, replay, trace

funsor.set_backend("torch")


class TraceEnum_ELBO(ELBO):

    def _get_trace(self, *args, **kwargs):
        raise ValueError("shouldn't be here")

    def differentiable_loss(self, model, guide, *args, **kwargs):
        # currently supports guide enumeration only, not mixed enumeration
        with trace() as guide_tr, enum(first_available_dim=-self.max_plate_nesting-1):
            guide(*args, **kwargs)
        with trace() as model_tr, replay(trace=guide_tr.trace):
            model(*args, **kwargs)
        factors, measures, measure_vars, plate_vars = [], [], frozenset(), frozenset()
        for role, tr in zip(("model", "guide"), (model_tr.trace, guide_tr.trace)):
            tr = prune_subsample_sites(tr)
            for name, node in tr.nodes.items():
                if role == "model":
                    factors.append(node["funsor"]["log_prob"])
                elif role == "guide":
                    factors.append(-node["funsor"]["log_prob"])
                    measures.append(node["funsor"]["log_measure"])
                plate_vars |= frozenset(f.name for f in node["cond_indep_stack"] if f.vectorized)
                measure_vars |= frozenset(node["funsor"]["log_prob"].inputs) - plate_vars

        # TODO support model enumeration
        # compute actual loss
        elbo = to_funsor(0., funsor.reals())
        # TODO get this optimizer call right
        with funsor.interpreter.interpretation(funsor.optimizer.optimize), funsor.memoize.memoize():
            for factor in factors:
                elbo = elbo + funsor.sum_product.sum_product(
                    funsor.ops.add, funsor.ops.mul,
                    [lm.exp() for lm in measures] + [factor],
                    eliminate=measure_vars | frozenset(factor.inputs) | plate_vars,
                    plates=plate_vars & frozenset(factor.inputs)
                )

        return -to_data(elbo)
