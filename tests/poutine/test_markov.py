# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from tests.common import assert_close


def check_inference(model, data):
    # Record a sequential trace.
    with poutine.trace() as tr:
        model(data)
    seq_trace = tr.trace
    print(list(seq_trace.nodes))

    # Stack samples.
    x = torch.stack([seq_trace.nodes["x_{}".format(t)]["value"]
                     for t in range(len(data))])

    # Record a vectorized trace.
    with poutine.trace() as tr:
        with poutine.condition(data={"x": x}):
            with poutine.vectorize_markov():  # TODO implement vectorize_markov
                model(data)
    vec_trace = tr.trace
    print(list(vec_trace.nodes))

    assert_close(seq_trace.log_prob_sum(),
                 vec_trace.log_prob_sum())


@pytest.mark.parametrize("duration", [0, 1, 2, 3, 4, 5])
def test_vectorized_function(duration):

    def model(data):
        def transition(state, t):
            x = state["x"]
            x = pyro.sample("x_{}".format(t), dist.Normal(0, 1))
            pyro.sample("y_{}".format(t), dist.Normal(x, 1),
                        obs=data[t])
            return {"x": x}

        # See pyro_reduce() below.
        x = pyro.sample("x_init", dist.Normal(0, 1))
        pyro.reduce(transition, range(len(data)), x)

    data = torch.randn(duration)
    check_inference(model, data)


@pytest.mark.parametrize("duration", [0, 1, 2, 3, 4, 5])
def test_vectorized_iterator(duration):

    def model(data):
        x = pyro.sample("x_init", dist.Normal(0, 1))
        for t in pyro.markov(range(len(data))):
            x = pyro.sample("x_{}".format(t), dist.Normal(x, 1))
            pyro.sample("y_{}".format(t), dist.Normal(x, 1),
                        obs=data[t])

    data = torch.randn(duration)
    check_inference(model, data)


################################################################################
# Sketch of autovectorization with an interface like functools.reduce(-,-,-)

_VECTORIZED = False


def pyro_reduce(transition, time, init):
    """
    This assumes ``init`` is a dict mapping unindexed sample site name (i.e.
    "x" rather than "x_0") to value.
    This assumes ``time`` is a range object.
    """
    if not _VECTORIZED:
        return functools.reduce(transition, time, init)

    # Trace the first step and assume model structure is fixed.
    # In pyro.contrib.epideiology we do this once at the start of inference.
    # Maybe we could memoize to avoid duplicated execution?
    with poutine.block(), poutine.trace() as tr:
        t = 0
        transition(init, t)
    names = [name for name in tr.trace.stochastic_nodes
             if name.endswith("_0")]

    # The remainder is vectorized over time.
    with pyro.plate("time", len(time)):
        t = slice(0, len(time), 1)

        # Record vectorized values.
        curr = {}
        prev = {}
        with poutine.block_trace_but_allow_replay_and_condition():
            for name in names:
                name_0 = "{}_{}".format(name, 0)
                name_t = "{}_{}".format(name, t)
                site_0 = tr.nodes[name_0]
                curr[name] = pyro.sample(name_t, site_0["fn"])
                prev[name] = torch.cat([site_0["value"].unsqueeze(0), curr[name]])

        # Execute vectorized transition.
        transition(prev, t)

    return {k: v[-1] for k, v in curr.items()}
