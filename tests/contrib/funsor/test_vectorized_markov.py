# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from torch.distributions import constraints

from pyro.ops.indexing import Vindex

# put all funsor-related imports here, so test collection works without funsor
try:
    import funsor
    import pyro.contrib.funsor
    from pyroapi import distributions as dist
    funsor.set_backend("torch")
    from pyroapi import handlers, pyro, pyro_backend
except ImportError:
    pytestmark = pytest.mark.skip(reason="funsor is not installed")


#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
def model_0(data, history, vectorized):
    x_dim = 3
    init = pyro.param("init", lambda: torch.rand(x_dim), constraint=constraints.simplex)
    trans = pyro.param("trans", lambda: torch.rand((x_dim, x_dim)), constraint=constraints.simplex)
    locs = pyro.param("locs", lambda: torch.rand(x_dim))

    with pyro.plate("sequences", data.shape[0], dim=-3) as sequences:
        sequences = sequences[:, None]
        x_prev = None
        markov_loop = \
            pyro.vectorized_markov(name="time", size=data.shape[1], dim=-2, history=history) if vectorized \
            else pyro.markov(range(data.shape[1]), history=history)
        for i in markov_loop:
            x_curr = pyro.sample(
                "x_{}".format(i), dist.Categorical(
                    init if isinstance(i, int) and i < 1 else trans[x_prev]),
                infer={"enumerate": "parallel"})
            with pyro.plate("tones", data.shape[2], dim=-1):
                pyro.sample("y_{}".format(i), dist.Normal(Vindex(locs)[..., x_curr], 1.),
                            obs=Vindex(data)[sequences, i])
            x_prev = x_curr


#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
def model_1(data, history, vectorized):
    x_dim = 3
    init = pyro.param("init", lambda: torch.rand(x_dim), constraint=constraints.simplex)
    trans = pyro.param("trans", lambda: torch.rand((x_dim, x_dim)), constraint=constraints.simplex)
    locs = pyro.param("locs", lambda: torch.rand(x_dim))

    x_prev = None
    markov_loop = \
        pyro.vectorized_markov(name="time", size=len(data), dim=-2, history=history) if vectorized \
        else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        x_curr = pyro.sample(
            "x_{}".format(i), dist.Categorical(
                init if isinstance(i, int) and i < 1 else trans[x_prev]),
            infer={"enumerate": "parallel"})
        with pyro.plate("tones", data.shape[-1], dim=-1):
            pyro.sample("y_{}".format(i), dist.Normal(Vindex(locs)[..., x_curr], 1.), obs=data[i])
        x_prev = x_curr


#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1] --> y[t] --> y[t+1]
def model_2(data, history, vectorized):
    x_dim, y_dim = 3, 2
    x_init = pyro.param("x_init", lambda: torch.rand(x_dim), constraint=constraints.simplex)
    x_trans = pyro.param("x_trans", lambda: torch.rand((x_dim, x_dim)), constraint=constraints.simplex)
    y_init = pyro.param("y_init", lambda: torch.rand(x_dim, y_dim), constraint=constraints.simplex)
    y_trans = pyro.param("y_trans", lambda: torch.rand((x_dim, y_dim, y_dim)), constraint=constraints.simplex)

    x_prev = y_prev = None
    markov_loop = \
        pyro.vectorized_markov(name="time", size=len(data), dim=-2, history=history) if vectorized \
        else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        x_curr = pyro.sample(
            "x_{}".format(i), dist.Categorical(
                x_init if isinstance(i, int) and i < 1 else x_trans[x_prev]),
            infer={"enumerate": "parallel"})
        with pyro.plate("tones", data.shape[-1], dim=-1):
            y_curr = pyro.sample("y_{}".format(i), dist.Categorical(
                y_init[x_curr] if isinstance(i, int) and i < 1 else Vindex(y_trans)[x_curr, y_prev]),
                obs=data[i])
        x_prev, y_prev = x_curr, y_curr


#    w[t-1] ----> w[t] ---> w[t+1]
#        \ x[t-1] --\-> x[t] --\-> x[t+1]
#         \  /       \  /       \  /
#          \/         \/         \/
#        y[t-1]      y[t]      y[t+1]
def model_3(data, history, vectorized):
    w_dim, x_dim, y_dim = 2, 3, 2
    w_init = pyro.param("w_init", lambda: torch.rand(w_dim), constraint=constraints.simplex)
    w_trans = pyro.param("w_trans", lambda: torch.rand((w_dim, w_dim)), constraint=constraints.simplex)
    x_init = pyro.param("x_init", lambda: torch.rand(x_dim), constraint=constraints.simplex)
    x_trans = pyro.param("x_trans", lambda: torch.rand((x_dim, x_dim)), constraint=constraints.simplex)
    y_probs = pyro.param("y_probs", lambda: torch.rand(w_dim, x_dim, y_dim), constraint=constraints.simplex)

    w_prev = x_prev = None
    markov_loop = \
        pyro.vectorized_markov(name="time", size=len(data), dim=-2, history=history) if vectorized \
        else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        w_curr = pyro.sample(
            "w_{}".format(i), dist.Categorical(
                w_init if isinstance(i, int) and i < 1 else w_trans[w_prev]),
            infer={"enumerate": "parallel"})
        x_curr = pyro.sample(
            "x_{}".format(i), dist.Categorical(
                x_init if isinstance(i, int) and i < 1 else x_trans[x_prev]),
            infer={"enumerate": "parallel"})
        with pyro.plate("tones", data.shape[-1], dim=-1):
            pyro.sample("y_{}".format(i), dist.Categorical(
                Vindex(y_probs)[w_curr, x_curr]),
                obs=data[i])
        x_prev, w_prev = x_curr, w_curr


#     w[t-1] ----> w[t] ---> w[t+1]
#        |  \       |  \       |   \
#        | x[t-1] ----> x[t] ----> x[t+1]
#        |   /      |   /      |   /
#        V  /       V  /       V  /
#     y[t-1]       y[t]      y[t+1]
def model_4(data, history, vectorized):
    w_dim, x_dim, y_dim = 2, 3, 2
    w_init = pyro.param("w_init", lambda: torch.rand(w_dim), constraint=constraints.simplex)
    w_trans = pyro.param("w_trans", lambda: torch.rand((w_dim, w_dim)), constraint=constraints.simplex)
    x_init = pyro.param("x_init", lambda: torch.rand(w_dim, x_dim), constraint=constraints.simplex)
    x_trans = pyro.param("x_trans", lambda: torch.rand((w_dim, x_dim, x_dim)), constraint=constraints.simplex)
    y_probs = pyro.param("y_probs", lambda: torch.rand(w_dim, x_dim, y_dim), constraint=constraints.simplex)

    w_prev = x_prev = None
    markov_loop = \
        pyro.vectorized_markov(name="time", size=len(data), dim=-2, history=history) if vectorized \
        else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        w_curr = pyro.sample(
            "w_{}".format(i), dist.Categorical(
                w_init if isinstance(i, int) and i < 1 else w_trans[w_prev]),
            infer={"enumerate": "parallel"})
        x_curr = pyro.sample(
            "x_{}".format(i), dist.Categorical(
                x_init[w_curr] if isinstance(i, int) and i < 1 else x_trans[w_curr, x_prev]),
            infer={"enumerate": "parallel"})
        with pyro.plate("tones", data.shape[-1], dim=-1):
            pyro.sample("y_{}".format(i), dist.Categorical(
                Vindex(y_probs)[w_curr, x_curr]),
                obs=data[i])
        x_prev, w_prev = x_curr, w_curr


#                     _______>______
#         _____>_____/______        \
#        /          /       \        \
#     x[t-1] --> x[t] --> x[t+1] --> x[t+2]
#        |        |          |          |
#        V        V          V          V
#     y[t-1]     y[t]     y[t+1]     y[t+2]
def model_5(data, history, vectorized):
    x_dim, y_dim = 3, 2
    x_init = pyro.param("x_init", lambda: torch.rand(x_dim), constraint=constraints.simplex)
    x_init_2 = pyro.param("x_init_2", lambda: torch.rand(x_dim, x_dim), constraint=constraints.simplex)
    x_trans = pyro.param("x_trans", lambda: torch.rand((x_dim, x_dim, x_dim)), constraint=constraints.simplex)
    y_probs = pyro.param("y_probs", lambda: torch.rand(x_dim, y_dim), constraint=constraints.simplex)

    x_prev = x_prev_2 = None
    markov_loop = \
        pyro.vectorized_markov(name="time", size=len(data), dim=-2, history=history) if vectorized \
        else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        if isinstance(i, int) and i == 0:
            x_probs = x_init
        elif isinstance(i, int) and i == 1:
            x_probs = Vindex(x_init_2)[x_prev]
        else:
            x_probs = Vindex(x_trans)[x_prev_2, x_prev]

        x_curr = pyro.sample(
            "x_{}".format(i), dist.Categorical(x_probs),
            infer={"enumerate": "parallel"})
        with pyro.plate("tones", data.shape[-1], dim=-1):
            pyro.sample("y_{}".format(i), dist.Categorical(
                Vindex(y_probs)[x_curr]),
                obs=data[i])
        x_prev_2, x_prev = x_prev, x_curr


# x_trans is time dependent
#
#     x[t-1] --> x[t] --> x[t+1]
#        |        |         |
#        V        V         V
#     y[t-1]     y[t]     y[t+1]
def model_6(data, history, vectorized):
    x_dim = 3
    x_init = pyro.param("x_init", lambda: torch.rand(x_dim), constraint=constraints.simplex)
    x_trans = pyro.param("x_trans", lambda: torch.rand((len(data)-1, x_dim, x_dim)), constraint=constraints.simplex)
    locs = pyro.param("locs", lambda: torch.rand(x_dim))

    x_prev = None
    markov_loop = \
        pyro.vectorized_markov(name="time", size=len(data), dim=-2, history=history) if vectorized \
        else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        if isinstance(i, int) and i < 1:
            x_probs = x_init
        elif isinstance(i, int):
            x_probs = x_trans[i-1, x_prev]
        else:
            x_probs = Vindex(x_trans)[(i-1)[:, None], x_prev]

        x_curr = pyro.sample(
            "x_{}".format(i), dist.Categorical(x_probs), infer={"enumerate": "parallel"})
        with pyro.plate("tones", data.shape[-1], dim=-1):
            pyro.sample("y_{}".format(i), dist.Normal(Vindex(locs)[..., x_curr], 1.), obs=data[i])
        x_prev = x_curr


#     w[t-1]      w[t]      w[t+1]
#        |  \    ^  | \    ^   |
#        |   \  /   |  \  /    |
#        v    \/    v   \/     v
#     y[t-1]  /\  y[t]  /\   y[t+1]
#        ^   /  \   ^  /  \    ^
#        |  /    v  | /    v   |
#     x[t-1]      x[t]      x[t+1]
def model_7(data, history, vectorized):
    w_dim, x_dim, y_dim = 2, 3, 2
    w_init = pyro.param("w_init", lambda: torch.rand(w_dim), constraint=constraints.simplex)
    w_trans = pyro.param("w_trans", lambda: torch.rand((x_dim, w_dim)), constraint=constraints.simplex)
    x_init = pyro.param("x_init", lambda: torch.rand(x_dim), constraint=constraints.simplex)
    x_trans = pyro.param("x_trans", lambda: torch.rand((w_dim, x_dim)), constraint=constraints.simplex)
    y_probs = pyro.param("y_probs", lambda: torch.rand(w_dim, x_dim, y_dim), constraint=constraints.simplex)

    w_prev = x_prev = None
    markov_loop = \
        pyro.vectorized_markov(name="time", size=len(data), dim=-2, history=history) if vectorized \
        else pyro.markov(range(len(data)), history=history)
    for i in markov_loop:
        w_curr = pyro.sample(
            "w_{}".format(i), dist.Categorical(
                w_init if isinstance(i, int) and i < 1 else w_trans[x_prev]),
            infer={"enumerate": "parallel"})
        x_curr = pyro.sample(
            "x_{}".format(i), dist.Categorical(
                x_init if isinstance(i, int) and i < 1 else x_trans[w_prev]),
            infer={"enumerate": "parallel"})
        with pyro.plate("tones", data.shape[-1], dim=-1):
            pyro.sample("y_{}".format(i), dist.Categorical(
                Vindex(y_probs)[w_curr, x_curr]),
                obs=data[i])
        x_prev, w_prev = x_curr, w_curr


@pytest.mark.parametrize("use_replay", [True, False])
@pytest.mark.parametrize("model,data,var,history", [
    (model_0, torch.rand(3, 5, 4), "xy", 1),
    (model_1, torch.rand(5, 4), "xy", 1),
    (model_2, torch.ones((5, 4), dtype=torch.long), "xy", 1),
    (model_3, torch.ones((5, 4), dtype=torch.long), "wxy", 1),
    (model_4, torch.ones((5, 4), dtype=torch.long), "wxy", 1),
    (model_5, torch.ones((5, 4), dtype=torch.long), "xy", 2),
    (model_6, torch.rand(5, 4), "xy", 1),
    (model_6, torch.rand(100, 4), "xy", 1),
    (model_7, torch.ones((5, 4), dtype=torch.long), "wxy", 1),
    (model_7, torch.ones((50, 4), dtype=torch.long), "wxy", 1),
])
def test_vectorized_markov(model, data, var, history, use_replay):

    with pyro_backend("contrib.funsor"), \
            handlers.enum():
        # sequential trace
        trace = handlers.trace(model).get_trace(data, history, False)

        # sequential factors
        factors = list()
        for i in range(data.shape[-2]):
            for v in var:
                factors.append(trace.nodes["{}_{}".format(v, i)]["funsor"]["log_prob"])

        # vectorized trace
        vectorized_trace = handlers.trace(model).get_trace(data, history, True)
        if use_replay:
            vectorized_trace = handlers.trace(
                    handlers.replay(model, trace=vectorized_trace)).get_trace(data, history, True)

        # vectorized factors
        vectorized_factors = list()
        for i in range(history):
            for v in var:
                vectorized_factors.append(vectorized_trace.nodes["{}_{}".format(v, i)]["funsor"]["log_prob"])
        for i in range(history, data.shape[-2]):
            for v in var:
                vectorized_factors.append(
                    vectorized_trace.nodes["{}_{}".format(v, slice(history, data.shape[-2]))]["funsor"]["log_prob"]
                    (**{"time": i-history},
                     **{"{}_{}".format(k, slice(history-j, data.shape[-2]-j)): "{}_{}".format(k, i-j)
                        for j in range(history+1) for k in var})
                    )

        # assert correct factors
        for f1, f2 in zip(factors, vectorized_factors):
            funsor.testing.assert_close(f2, f1.align(tuple(f2.inputs)))

        # assert correct step
        actual_step = vectorized_trace.nodes["time"]["value"]
        # expected step: assume that all but the last var is markov
        expected_step = frozenset()
        for v in var[:-1]:
            v_step = tuple("{}_{}".format(v, i) for i in range(history)) \
                     + tuple("{}_{}".format(v, slice(j, data.shape[-2]-history+j)) for j in range(history+1))
            expected_step |= frozenset({v_step})
        assert actual_step == expected_step


#     x[i-1] --> x[i] --> x[i+1]
#        |        |         |
#        V        V         V
#     y[i-1]     y[i]     y[i+1]
#
#     w[j-1] --> w[j] --> w[j+1]
#        |        |         |
#        V        V         V
#     z[j-1]     z[j]     z[j+1]
def model_8(weeks_data, days_data, history, vectorized):
    x_dim, y_dim, w_dim, z_dim = 3, 2, 2, 3
    x_init = pyro.param("x_init", lambda: torch.rand(x_dim), constraint=constraints.simplex)
    x_trans = pyro.param("x_trans", lambda: torch.rand((x_dim, x_dim)), constraint=constraints.simplex)
    y_probs = pyro.param("y_probs", lambda: torch.rand(x_dim, y_dim), constraint=constraints.simplex)
    w_init = pyro.param("w_init", lambda: torch.rand(w_dim), constraint=constraints.simplex)
    w_trans = pyro.param("w_trans", lambda: torch.rand((w_dim, w_dim)), constraint=constraints.simplex)
    z_probs = pyro.param("z_probs", lambda: torch.rand(w_dim, z_dim), constraint=constraints.simplex)

    x_prev = None
    weeks_loop = \
        pyro.vectorized_markov(name="weeks", size=len(weeks_data), dim=-1, history=history) if vectorized \
        else pyro.markov(range(len(weeks_data)), history=history)
    for i in weeks_loop:
        if isinstance(i, int) and i == 0:
            x_probs = x_init
        else:
            x_probs = Vindex(x_trans)[x_prev]

        x_curr = pyro.sample(
            "x_{}".format(i), dist.Categorical(x_probs), infer={"enumerate": "parallel"})
        pyro.sample("y_{}".format(i), dist.Categorical(Vindex(y_probs)[x_curr]), obs=weeks_data[i])
        x_prev = x_curr

    w_prev = None
    days_loop = \
        pyro.vectorized_markov(name="days", size=len(days_data), dim=-1, history=history) if vectorized \
        else pyro.markov(range(len(days_data)), history=history)
    for j in days_loop:
        if isinstance(j, int) and j == 0:
            w_probs = w_init
        else:
            w_probs = Vindex(w_trans)[w_prev]

        w_curr = pyro.sample(
            "w_{}".format(j), dist.Categorical(w_probs), infer={"enumerate": "parallel"})
        pyro.sample("z_{}".format(j), dist.Categorical(Vindex(z_probs)[w_curr]), obs=days_data[j])
        w_prev = w_curr


@pytest.mark.parametrize("use_replay", [True, False])
@pytest.mark.parametrize("model,weeks_data,days_data,vars1,vars2,history", [
    (model_8, torch.ones(3), torch.zeros(9), "xy", "wz", 1),
    (model_8, torch.ones(30), torch.zeros(50), "xy", "wz", 1),
])
def test_vectorized_markov_multi(model, weeks_data, days_data, vars1, vars2, history, use_replay):

    with pyro_backend("contrib.funsor"), \
            handlers.enum():
        # sequential factors
        trace = handlers.trace(model).get_trace(weeks_data, days_data, history, False)

        factors = list()
        # sequential weeks factors
        for i in range(len(weeks_data)):
            for v in vars1:
                factors.append(trace.nodes["{}_{}".format(v, i)]["funsor"]["log_prob"])
        # sequential days factors
        for j in range(len(days_data)):
            for v in vars2:
                factors.append(trace.nodes["{}_{}".format(v, j)]["funsor"]["log_prob"])

        # vectorized trace
        vectorized_trace = handlers.trace(model).get_trace(weeks_data, days_data, history, True)
        if use_replay:
            vectorized_trace = handlers.trace(
                    handlers.replay(model, trace=vectorized_trace)).get_trace(weeks_data, days_data, history, True)

        vectorized_factors = list()
        # vectorized weeks factors
        for i in range(history):
            for v in vars1:
                vectorized_factors.append(vectorized_trace.nodes["{}_{}".format(v, i)]["funsor"]["log_prob"])
        for i in range(history, len(weeks_data)):
            for v in vars1:
                vectorized_factors.append(
                    vectorized_trace.nodes[
                        "{}_{}".format(v, slice(history, len(weeks_data)))]["funsor"]["log_prob"]
                    (**{"weeks": i-history},
                     **{"{}_{}".format(k, slice(history-j, len(weeks_data)-j)): "{}_{}".format(k, i-j)
                        for j in range(history+1) for k in vars1})
                    )
        # vectorized days factors
        for i in range(history):
            for v in vars2:
                vectorized_factors.append(vectorized_trace.nodes["{}_{}".format(v, i)]["funsor"]["log_prob"])
        for i in range(history, len(days_data)):
            for v in vars2:
                vectorized_factors.append(
                    vectorized_trace.nodes[
                        "{}_{}".format(v, slice(history, len(days_data)))]["funsor"]["log_prob"]
                    (**{"days": i-history},
                     **{"{}_{}".format(k, slice(history-j, len(days_data)-j)): "{}_{}".format(k, i-j)
                        for j in range(history+1) for k in vars2})
                    )

        # assert correct factors
        for f1, f2 in zip(factors, vectorized_factors):
            funsor.testing.assert_close(f2, f1.align(tuple(f2.inputs)))

        # assert correct step

        actual_weeks_step = vectorized_trace.nodes["weeks"]["value"]
        # expected step: assume that all but the last var is markov
        expected_weeks_step = frozenset()
        for v in vars1[:-1]:
            v_step = tuple("{}_{}".format(v, i) for i in range(history)) \
                     + tuple("{}_{}".format(v, slice(j, len(weeks_data)-history+j)) for j in range(history+1))
            expected_weeks_step |= frozenset({v_step})

        actual_days_step = vectorized_trace.nodes["days"]["value"]
        # expected step: assume that all but the last var is markov
        expected_days_step = frozenset()
        for v in vars2[:-1]:
            v_step = tuple("{}_{}".format(v, i) for i in range(history)) \
                     + tuple("{}_{}".format(v, slice(j, len(days_data)-history+j)) for j in range(history+1))
            expected_days_step |= frozenset({v_step})

        assert actual_weeks_step == expected_weeks_step
        assert actual_days_step == expected_days_step
