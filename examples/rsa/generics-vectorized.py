#!/usr/bin/env python3

# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# %%
import argparse
import collections
import numbers

import torch
from search_inference import HashingMarginal, Search, memoize

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import config_enumerate, TraceEnum_ELBO
from pyro.ops.indexing import Vindex

torch.set_default_dtype(torch.float64)  # double precision for numerical stability
torch.manual_seed(42)

utterances = [
    "generic is true", "generic is false",
    "mu", "some", "most", "all",
]

from vectorized_search import (
    VectoredSearch as VSearch,
    VectoredHashingMarginal as VHMarginal,
)


def Marginal(fn):
    return memoize(lambda *args: VHMarginal(VSearch(config_enumerate(fn)).run(*args)))


Params = collections.namedtuple("Params", ["theta", "gamma", "delta"])
beta_bins = torch.tensor([0., 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])


# %%
@Marginal
def structured_prior(params: Params) -> torch.Tensor:
    # computing the Beta pdf for discretized bins above for enumerated Search
    shape_alpha = params.gamma * params.delta - 1
    shape_beta  = (1. - params.gamma) * params.delta - 1
    discrete_bins = (beta_bins ** shape_alpha) * ((1. - beta_bins) ** shape_beta) * params.theta
    discrete_bins[0] = (1 - params.theta)
    idx = pyro.sample("bin", dist.Categorical(probs=discrete_bins / discrete_bins.sum()))

    return beta_bins[idx]


# %%
wings_prior_params = Params(theta=0.5, gamma=0.99, delta=10.0)
wings_prior = structured_prior(wings_prior_params)

for el in wings_prior.enumerate_support():
    print(el.item(), wings_prior.log_prob(el).exp().item())


# %%
def utterance_prior() -> torch.Tensor:
    utts = torch.arange(0, len(utterances), 1)
    probs = torch.ones_like(utts) / len(utts)
    idx = pyro.sample("utterance", dist.Categorical(probs=probs))
    return utts[idx]


# %%
def threshold_prior() -> torch.Tensor:
    bins = torch.arange(0.0, 1.0, 0.1)
    idx = pyro.sample("threshold", dist.Categorical(logits=torch.zeros_like(bins)))
    return bins[idx]


# %%
def meaning(utterance: torch.Tensor, state: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    possible_evals = {
        "as_genT": (state > threshold),
        "as_genF": (state <= threshold),
        "is_mu"  : torch.full_like(state, True, dtype=bool),
        "is_some": (state > 0),
        "is_most": (state >= 0.5),
        "is_all" : (state >= 0.99),
        "as_num" : (state == utterance),
        "default": torch.full_like(state, True, dtype=bool),
    }

    meanings = torch.stack(list(possible_evals.values()))

    while utterance.ndim < meanings.ndim:  # expand utterance to be used as an indexer
        utterance = utterance[None]

    return torch.gather(meanings, dim=0, index=utterance.long()).float().squeeze()

# %% [markdown]
# # Listener 0

# %%
@Marginal
def listener0(utterances: torch.Tensor, thresholds: torch.Tensor, prior: HashingMarginal) -> torch.Tensor:
    state = pyro.sample(f"state", prior)
    means = meaning(utterances, state, thresholds)
    pyro.factor(f"listener0-true", torch.where(means == 1., 0., -99_999.))
    return state


# %%
wings_posterior = listener0(torch.tensor([1]), torch.tensor([0.1]), wings_prior)
for el in wings_posterior.enumerate_support():
    print(el, wings_posterior.log_prob(el).exp().item())


