from __future__ import absolute_import, division, print_function

import logging

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim
from pyro.infer.search import Search, ParallelSearch
from tests.common import assert_equal

logger = logging.getLogger(__name__)


def test_elbo_hmm_in_model():
    pyro.clear_param_store()
    data = torch.ones(3)
    init_probs = torch.tensor([0.5, 0.5])

    def model(data):
        transition_probs = pyro.param("transition_probs",
                                      torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
                                      constraint=constraints.simplex)
        locs = pyro.param("obs_locs", torch.tensor([-1, 1]))
        scale = pyro.param("obs_scale", torch.tensor(1.0),
                           constraint=constraints.positive)

        x = None
        for i, y in enumerate(data):
            probs = init_probs if x is None else transition_probs[x]
            x = pyro.sample("x_{}".format(i), dist.Categorical(probs))
            pyro.sample("y_{}".format(i), dist.Normal(locs[x], scale), obs=y)

    # XXX max_iarange_nesting = 0 ?
    sequential_posterior = Search(model)
    parallel_posterior = ParallelSearch(model)

    parallel_trace = [(tr, log_w) for tr, log_w in parallel_posterior._traces(data)]
    seq_traces = [(tr, log_w) for tr, log_w in sequential_posterior._traces(data)]
