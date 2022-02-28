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

    @ConditionMessenger(data={f"{n}": value for n in range(NGaussians + 1)})
    def model():
        with pyro.plate("z_plate", 2):
            with pyro.plate("samples", S):
                for n in range(NGaussians + 1):
                    x = pyro.sample(
                        f"z_{n}", dist.Normal(0, 1)
                    )  # .expand([10]) is automatic
                    assert x.shape == torch.Size([S, 2])

    for _ in range(B):
        trace = pyro.poutine.trace(model).get_trace()  # type: ignore
        trace.add_node(LW, name=LW, type="return", value=lw)
        breakpoint()
        resampled = resample_trace_systematic(trace, sample_dims=0, batch_dim=None)
        assert (resampled.nodes["_LOGWEIGHT"]["value"].exp() == 0.25).all()
        breakpoint()
        for k, site in resampled.nodes.items():
            if k[0:2] == "z_":
                for s in range(S):
                    memo[s] += (site["value"] == (s + 1)).sum() / (S * NGaussians * 2)
        breakpoint()
    print(memo / B)


# def test_resample_without_batch():
#    S = 4
#    N = 5
#    B = 100
#
#    value = torch.tensor([[1,1],[2,2],[3,3],[4,4]])
#    lw = torch.tensor([0.1, 0.2, 0.3, 0.4]).log()
#    tr = Trace()
#
#    memo = torch.zeros(S)
#    for _ in range(B):
#        for n in range(N):
#            tr._inject(RandomVariable(dist=D.Normal(0, 1), value=value, log_prob=lw, reparameterized=False), name=f'z_{n}')
#
#        resampled, _lw = Systematic()(tr, lw, sample_dims=0, batch_dim=None)
#
#        assert (_lw.exp() == 0.25).all()
#        for n, (_, rv) in enumerate(resampled.items()):
#            for s in range(S):
#                memo[s] += (rv.value == (s+1)).sum() / (S*N*2)
#
#    print(memo / B)


@mark.skip()
def test_resample_with_batch():
    N, S, B = 5, 4, 100
    init = torch.arange(1, 5)

    value = init.expand(2, S).T
    lw = (init / 10).log()
    breakpoint()

    lw = lw.unsqueeze(1).expand(S, B)
    value = value.unsqueeze(1).expand(S, B, 2)

    tr = Trace()

    for n in range(N):

        tr._inject(
            RandomVariable(
                dist=D.Normal(0, 1), value=value, log_prob=lw, reparameterized=False
            ),
            name=f"z_{n}",
            silent=True,
        )

    resampled, _lw = Systematic()(tr, lw, sample_dims=0, batch_dim=1)
    assert (_lw.exp() == 0.25).all()

    memo = torch.zeros(S)
    for n, (_, rv) in enumerate(resampled.items()):
        for s in range(S):
            memo[s] += (rv.value == (s + 1)).sum() / (S * B * N * 2)

    print(memo)
