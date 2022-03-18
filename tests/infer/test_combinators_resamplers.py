#!/usr/bin/env python3
import pyro
import pyro.distributions as dist
import torch

from pytest import mark
from pyro.contrib.funsor.handlers.trace_messenger import TraceMessenger
from pyro.poutine.condition_messenger import ConditionMessenger
from pyro.infer.combinators import (
    ancestor_indices_systematic,
    resample_trace_systematic,
)


def test_ancestor_indices_systematic():
    S = 4
    B = 1000
    init = torch.tensor([0.1, 0.2, 0.3, 0.4])
    ret = torch.empty_like(init)

    lw = init.log()
    lw = lw.unsqueeze(1).expand(S, B)
    a = ancestor_indices_systematic(lw, 0, 1).T
    for i in range(S):
        ret[i] = (a == i).sum() / (S * B)

    # don't go all crazy with the batch size here
    assert torch.isclose(init, ret, atol=1e-3).all()


def test_resample_without_batch():
    S = 4
    NGaussians = 4
    B = 100

    LW = "_LOGWEIGHT"
    init = torch.arange(1, NGaussians + 1)
    value = init.expand(2, S).T
    lw = (init / 10).log()
    assert value.shape == torch.Size([S, 2])

    memo = torch.zeros(S)

    @ConditionMessenger(data={f"z_{n}": value for n in range(NGaussians + 1)})
    def model():
        with pyro.plate("plate", 2):
            with pyro.plate("samples", S):
                for n in range(NGaussians):
                    x = pyro.sample(
                        f"z_{n}", dist.Normal(0, 1)
                    )  # .expand([10]) is automatic
                    assert x.shape == torch.Size([S, 2])

    for _ in range(B):
        trace = pyro.poutine.trace(model).get_trace()  # type: ignore
        trace.add_node(LW, name=LW, type="return", value=lw)
        resampled = resample_trace_systematic(trace, sample_dims=0, batch_dim=None)
        assert torch.isclose(
            resampled.nodes["_LOGWEIGHT"]["value"].exp(), torch.ones(4) * 0.25
        ).all()
        for k, site in resampled.nodes.items():
            if k[0:2] == "z_":
                for s in range(S):
                    memo[s] += (site["value"] == (s + 1)).sum() / (S * NGaussians * 2)

    assert torch.isclose((memo / B), torch.ones(4) * 0.25).all()


def test_resample_with_batch():
    NGaussians, S, B = 5, 4, 100
    init = torch.arange(1, 5)
    value = init.expand(2, S).T
    LW = "_LOGWEIGHT"
    lw = (init / 10).log()

    lw = lw.unsqueeze(1).expand(S, B)
    value = value.unsqueeze(1).expand(S, B, 2)

    @ConditionMessenger(data={f"z_{n}": value for n in range(NGaussians + 1)})
    def model():
        with pyro.plate("plate", 2):
            with pyro.plate("samples", S):
                with pyro.plate("batch", B):
                    for n in range(NGaussians):
                        x = pyro.sample(f"z_{n}", dist.Normal(0, 1))
                        assert x.shape == torch.Size([S, B, 2])

    trace = pyro.poutine.trace(model).get_trace()  # type: ignore
    trace.add_node(LW, name=LW, type="return", value=lw)

    resampled = resample_trace_systematic(trace, sample_dims=0, batch_dim=1)
    assert torch.isclose(
        resampled.nodes["_LOGWEIGHT"]["value"].exp(), torch.ones(S, B) * 0.25
    ).all()

    memo = torch.zeros(S)
    for k, site in resampled.nodes.items():
        if k[0:2] == "z_":
            for s in range(S):
                memo[s] += (site["value"] == (s + 1)).sum() / (S * B * NGaussians * 2)

    assert torch.isclose((memo), torch.ones(4) * 0.25).all()


def test_resample_with_outputs():
    S, NGaussians, B = 4, 4, 100
    LW = "_LOGWEIGHT"
    init = torch.arange(1, NGaussians + 1)
    value = init.expand(S).T
    lw = (init / 10).log()
    assert value.shape == torch.Size([S])
    as_set = lambda out: {v.item() for v in out}

    def model_one():
        with pyro.plate("samples", S):
            return pyro.sample(f"z_0", dist.Normal(0, 1))

    trace = pyro.poutine.trace(model_one).get_trace()  # type: ignore
    trace.add_node(LW, name=LW, type="return", value=torch.arange(S).float())
    out = trace.nodes["_RETURN"]["value"]
    resampled = resample_trace_systematic(trace, sample_dims=0, batch_dim=None)
    resampled_out = resampled.nodes["_RETURN"]["value"]

    assert (
        len(as_set(out) - as_set(resampled_out)) > 0
        and resampled_out is resampled.nodes["z_0"]["value"]
    )

    def model_list():
        with pyro.plate("samples", S):
            ret = []
            for n in range(NGaussians):
                ret.append(pyro.sample(f"z_{n}", dist.Normal(0, 1)))
            return ret

    trace = pyro.poutine.trace(model_list).get_trace()  # type: ignore
    trace.add_node(LW, name=LW, type="return", value=torch.arange(S).float())
    out = [x.clone() for x in trace.nodes["_RETURN"]["value"]]
    resampled = resample_trace_systematic(trace, sample_dims=0, batch_dim=None)
    rout = [x.clone() for x in resampled.nodes["_RETURN"]["value"]]
    for i, (o, r) in enumerate(zip(out, rout)):
        assert (
            len(as_set(o) - as_set(r)) > 0
            and (r == resampled.nodes[f"z_{i}"]["value"]).all()
        )

    def model_dict():
        with pyro.plate("samples", S):
            ret = dict()
            for n in range(NGaussians):
                ret[n] = pyro.sample(f"z_{n}", dist.Normal(0, 1))
            return ret

    trace = pyro.poutine.trace(model_dict).get_trace()  # type: ignore
    trace.add_node(LW, name=LW, type="return", value=torch.arange(S).float())
    out = {i: x.clone() for i, x in trace.nodes["_RETURN"]["value"].items()}
    resampled = resample_trace_systematic(trace, sample_dims=0, batch_dim=None)
    rout = {i: x.clone() for i, x in resampled.nodes["_RETURN"]["value"].items()}

    for (oix, o), (rix, r) in zip(out.items(), rout.items()):
        assert (
            oix == rix
            and len(as_set(o) - as_set(r)) > 0
            and (r == resampled.nodes[f"z_{oix}"]["value"]).all()
        )
