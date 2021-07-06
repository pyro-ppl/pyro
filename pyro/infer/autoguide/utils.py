# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

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
    entropy = 0.0
    for name, site in trace.nodes.items():
        if site["type"] == "sample":
            if not poutine.util.site_is_subsample(site):
                if whitelist is None or name in whitelist:
                    entropy += site["fn"].entropy()
    return entropy


@contextmanager
def helpful_support_errors(site):
    try:
        yield
    except NotImplementedError as e:
        support_name = repr(site["fn"].support).lower()
        if site["fn"].support.is_discrete:
            name = site["name"]
            raise ValueError(
                f"Continuous inference cannot handle discrete sample site '{name}'. "
                "Consider enumerating that variable as documented in "
                "https://pyro.ai/examples/enumeration.html . "
                "If you are already enumerating, take care to hide this site when "
                "constructing an autoguide, e.g. "
                f"guide = AutoNormal(poutine.block(model, hide=['{name}']))."
            )
        if "sphere" in support_name:
            name = site["name"]
            raise ValueError(
                f"Continuous inference cannot handle spherical sample site '{name}'. "
                "Consider using ProjectedNormal distribution together with "
                "a reparameterizer, e.g. "
                f"poutine.reparam(config={{'{name}': ProjectedNormalReparam()}})."
            )
        raise e from None
