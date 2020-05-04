# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import funsor

from pyro.infer import ELBO
from pyro.poutine.util import prune_subsample_sites

from pyro.contrib.funsor import to_data
from pyro.contrib.funsor.handlers import enum, replay, trace

funsor.set_backend("torch")


class TraceTMC_ELBO(ELBO):

    def _get_trace(self, *args, **kwargs):
        raise ValueError("shouldn't be here")

    def differentiable_loss(self, model, guide, *args, **kwargs):
        with enum(first_available_dim=-self.max_plate_nesting-1):
            guide_tr = trace(guide).get_trace(*args, **kwargs)
            model_tr = trace(replay(model, trace=guide_tr)).get_trace(*args, **kwargs)

        log_factors, log_measures, measure_vars, plate_vars = [], [], frozenset(), frozenset()
        for role, tr in zip(("model", "guide"), map(prune_subsample_sites, (model_tr, guide_tr))):
            for name, node in tr.nodes.items():
                if node["type"] != "sample":
                    continue
                log_factors.append(
                    node["funsor"]["log_prob"] if role == "model" else -node["funsor"]["log_prob"])
                if node["funsor"].get("log_measure", None) is not None:
                    log_measures.append(node["funsor"]["log_measure"])
                    measure_vars |= frozenset(node["funsor"]["log_measure"].inputs)
                plate_vars |= frozenset(f.name for f in node["cond_indep_stack"] if f.vectorized)
                measure_vars |= frozenset(node["funsor"]["log_prob"].inputs)

        with funsor.interpreter.interpretation(funsor.terms.lazy):
            elbo = funsor.sum_product.sum_product(
                funsor.ops.logaddexp, funsor.ops.add, log_measures + log_factors,
                eliminate=measure_vars | plate_vars, plates=plate_vars
            )
        return -to_data(funsor.optimizer.apply_optimizer(elbo))
