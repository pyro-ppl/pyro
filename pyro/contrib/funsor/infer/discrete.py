# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import funsor

from pyro.contrib.funsor.handlers import enum, replay, trace
from pyro.contrib.funsor.handlers.enum_messenger import _get_support_value
from pyro.contrib.funsor.infer.traceenum_elbo import terms_from_trace
from pyro.poutine import Trace, block
from pyro.poutine.util import site_is_subsample


def _sample_posterior(model, first_available_dim, temperature, *args, **kwargs):

    if temperature == 0:
        sum_op, prod_op = funsor.ops.max, funsor.ops.add
        approx = funsor.approximations.argmax_approximate
    elif temperature == 1:
        sum_op, prod_op = funsor.ops.logaddexp, funsor.ops.add
        approx = funsor.montecarlo.MonteCarlo()
    else:
        raise ValueError("temperature must be 0 (map) or 1 (sample) for now")

    with block(), enum(first_available_dim=first_available_dim):
        # XXX replay against an empty Trace to ensure densities are not double-counted
        model_tr = trace(replay(model, trace=Trace())).get_trace(*args, **kwargs)

    terms = terms_from_trace(model_tr)
    # terms["log_factors"] = [log p(x) for each observed or latent sample site x]
    # terms["log_measures"] = [log p(z) or other Dice factor
    #                          for each latent sample site z]

    with funsor.interpretations.lazy:
        log_prob = funsor.sum_product.sum_product(
            sum_op,
            prod_op,
            terms["log_factors"] + terms["log_measures"],
            eliminate=terms["measure_vars"] | terms["plate_vars"],
            plates=terms["plate_vars"],
        )
        log_prob = funsor.optimizer.apply_optimizer(log_prob)

    with approx:
        approx_factors = funsor.adjoint.adjoint(sum_op, prod_op, log_prob)

    # construct a result trace to replay against the model
    sample_tr = model_tr.copy()
    sample_subs = {}
    for name, node in sample_tr.nodes.items():
        if node["type"] != "sample" or site_is_subsample(node):
            continue
        if node["is_observed"]:
            # "observed" values may be collapsed samples that depend on enumerated
            # values, so we have to slice them down
            # TODO this should really be handled entirely under the hood by adjoint
            node["funsor"] = {"value": node["funsor"]["value"](**sample_subs)}
        else:
            node["funsor"]["log_measure"] = approx_factors[
                node["funsor"]["log_measure"]
            ]
            node["funsor"]["value"] = _get_support_value(
                node["funsor"]["log_measure"], name
            )
            sample_subs[name] = node["funsor"]["value"]

    with replay(trace=sample_tr):
        return model(*args, **kwargs)


def infer_discrete(model, first_available_dim=None, temperature=1):
    return functools.partial(_sample_posterior, model, first_available_dim, temperature)
