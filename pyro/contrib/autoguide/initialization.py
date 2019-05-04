from __future__ import absolute_import, division, print_function

import torch
from torch.distributions import transform_to


@torch.no_grad()
def init_to_feasible(site):
    value = site["value"].detach()
    t = transform_to(site["fn"].support)
    return t(torch.zeros_like(t.inv(value)))


@torch.no_grad()
def init_to_sample(site):
    return site["value"].detach()


@torch.no_grad()
def init_to_mean(site):
    try:
        value = site["fn"].mean.detach()
        if hasattr(site["fn"], "_validate_sample"):
            # Fall back to a feasible point for distributions with
            # infinite variance, e.g. Cauchy.
            site["fn"]._validate_sample(value)
        return value
    except (NotImplementedError, ValueError):
        return init_to_feasible(site)


@torch.no_grad()
def init_to_median(site, num_samples=32):
    samples = site["fn"].sample(sample_shape=(num_samples,))
    return samples.median(dim=0)[0]
