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


def test_vectorized_markov():
    data = torch.ones(5)

    pyro.get_param_store().clear()
    init = pyro.param("init", lambda: torch.rand(3), constraint=constraints.simplex)
    trans = pyro.param("trans", lambda: torch.rand((3, 3)), constraint=constraints.simplex)
    locs = pyro.param("locs", lambda: torch.rand((3,)))

    def model():
        for i in pyro.markov(range(len(data))):
            x_curr = pyro.sample(
                "x_{}".format(i), dist.Categorical(
                    init if isinstance(i, int) and i < 1 else trans[x_prev]),
                infer={"enumerate": "parallel"})
            pyro.sample("y_{}".format(i), dist.Normal(Vindex(locs)[..., x_curr], 1.), obs=data[i])
            x_prev = x_curr

    def vectorized_model():
        for i in pyro.vectorized_markov(name="time", size=len(data), dim=-1):
            x_curr = pyro.sample(
                "x_{}".format(i), dist.Categorical(
                    init if isinstance(i, int) and i < 1 else trans[x_prev]),
                infer={"enumerate": "parallel"})
            pyro.sample("y_{}".format(i), dist.Normal(Vindex(locs)[..., x_curr], 1.), obs=data[i])
            x_prev = x_curr

    with pyro_backend("contrib.funsor"), \
            handlers.enum():
        trace = handlers.trace(model).get_trace()
        factors = list()
        for i in range(len(data)):
            factors.append(trace.nodes["x_{}".format(i)]["funsor"]["log_prob"])
            factors.append(trace.nodes["y_{}".format(i)]["funsor"]["log_prob"])

        vectorized_trace = handlers.trace(vectorized_model).get_trace()
        vectorized_factors = list()
        vectorized_factors.append(vectorized_trace.nodes["x_0"]["funsor"]["log_prob"])
        vectorized_factors.append(vectorized_trace.nodes["y_0"]["funsor"]["log_prob"])
        for i in range(1, len(data)):
            vectorized_factors.append(
                vectorized_trace.nodes["x_{}".format(torch.arange(1, 5))]["funsor"]["log_prob"]
                (**{"time": i-1,
                    "x_{}".format(torch.arange(1, 5)): "x_{}".format(i),
                    "x_{}".format(torch.arange(4)): "x_{}".format(i-1)})
                )
            vectorized_factors.append(
                vectorized_trace.nodes["y_{}".format(torch.arange(1, 5))]["funsor"]["log_prob"]
                (**{"time": i-1,
                    "x_{}".format(torch.arange(1, 5)): "x_{}".format(i),
                    "y_{}".format(torch.arange(1, 5)): "y_{}".format(i)})
                )

        for f1, f2 in zip(factors, vectorized_factors):
            assert set(f1.inputs) == set(f2.inputs)
            assert torch.equal(pyro.to_data(f1), pyro.to_data(f2))
