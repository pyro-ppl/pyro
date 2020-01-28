# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from pyro import poutine


def _product(shape):
    """
    Computes the product of the dimensions of a given shape tensor
    """
    result = 1
    for size in shape:
        result *= size
    return result


def mean_field_entropy(model, args, whitelist=None):
    """Computes the entropy of a model, assuming
    that the model is fully mean-field (i.e. all sample sites
    in the model are independent).

    The entropy is simply the sum of the entropies at the
    individual sites. If `whitelist` is not `None`, only sites
    listed in `whitelist` will have their entropies included
    in the sum. If `whitelist` is `None`, all non-subsample
    sites are included.
    """
    trace = poutine.trace(model).get_trace(*args)
    entropy = 0.
    for name, site in trace.nodes.items():
        if site["type"] == "sample":
            if not poutine.util.site_is_subsample(site):
                if whitelist is None or name in whitelist:
                    entropy += site["fn"].entropy()
    return entropy
