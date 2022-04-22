# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro.infer.combinators as combinator
from functools import reduce
from pytest import mark, fixture
from torch import Tensor, tensor
from typing import Any, Callable, Optional, Union, Optional

import pyro
import pyro.distributions as dist
import pyro.optim.pytorch_optimizers as optim
from pyro import poutine
from pyro.infer.combinators import (
    compose,
    extend,
    primitive,
    propose,
    with_substitution,
    is_auxiliary,
    is_sample_type,
    stl_trick,
    augment_logweight,
    nested_objective,
    _LOSS,
    _RETURN,
)
from pyro.nn.module import PyroModule
from pyro.poutine import Trace, replay
from pyro.poutine.handlers import _make_handler
from pyro.poutine.messenger import Messenger
from pyro.poutine.trace_messenger import TraceMessenger


def seed(s=42) -> None:
    import random

    import numpy as np

    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)
    # just incase something goes wrong with set_deterministic
    torch.backends.cudnn.benchmark = True
    if torch.__version__[:3] == "1.8":
        pass
        # torch.use_deterministic_algorithms(True)


def thash(
    aten: Tensor,
    length: int = 8,
    with_ref=False,
    no_grad_char: str = " ",
) -> str:
    import pickle
    import base64
    import hashlib

    def _hash(t: Tensor, length: int) -> str:
        hasher = hashlib.sha1(pickle.dumps(t))
        return base64.urlsafe_b64encode(hasher.digest()[:length]).decode("ascii")

    g = "∇" if aten.grad_fn is not None else no_grad_char
    save_ref = aten.detach()
    if with_ref:
        r = _hash(save_ref, (length // 4))
        v = _hash(save_ref.cpu().numpy(), 3 * (length // 4))
        return f"#{g}{r}{v}"
    else:
        v = _hash(save_ref.cpu().numpy(), length)
        return f"#{g}{v}"


class LinearKernel(PyroModule):
    """simple MLP kernel that expects an observation with a sample dimension"""

    def __init__(self, name, dim=1):
        super().__init__()
        self.name = name
        self.net = nn.Linear(dim, dim)

    def forward(self, obs):
        return pyro.sample(
            self.name, dist.Normal(loc=self.net(obs.T.detach()).T, scale=1.0)
        )


@pytest.fixture(scope="session", autouse=True)
def linear_kernel():
    def model(name):
        return primitive(LinearKernel(name, dim=1))

    yield model


class MLPKernel(PyroModule):
    """simple MLP kernel that expects an observation with a sample dimension"""

    def __init__(self, name, dim_hidden):
        super().__init__()
        self.name = name
        self.net = nn.Sequential(
            nn.Linear(1, dim_hidden),
            nn.Sigmoid(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Sigmoid(),
            nn.Linear(dim_hidden, 1),
        )

    def forward(self, obs):
        return pyro.sample(
            self.name, dist.Normal(loc=self.net(obs.T.detach()).T, scale=1.0)
        )

    def __repr__(self):
        return f"MLPKernel({self.name})"


@pytest.fixture(scope="session", autouse=True)
def mlp_kernel():
    def model(name, dim_hidden):
        return primitive(MLPKernel(name, dim_hidden=dim_hidden))

    yield model


class Normal(PyroModule):
    def __init__(self, name, loc=0, scale=1):
        super().__init__()
        self.name, self.loc, self.scale = name, loc, scale

    def forward(self):
        return pyro.sample(
            self.name,
            dist.Normal(
                loc=torch.ones(1, 1) * self.loc, scale=torch.ones(1, 1) * self.scale
            ),
        )

    def __repr__(self):
        return f"Normal({self.name})"


@pytest.fixture(scope="session", autouse=True)
def normal():
    def model(name: str, loc: int = 0, scale: int = 1):
        return primitive(Normal(name, loc, scale))

    yield model


@nested_objective
def nvo_avo(p_out, q_out, lw, lv, sample_dims=-1) -> Tensor:
    nw = F.softmax(torch.zeros_like(lv), dim=sample_dims)
    loss = (nw * (-lv)).sum(dim=(sample_dims,), keepdim=False)
    return loss


def test_gradient_updates(normal, mlp_kernel):
    q = normal("z_0", loc=4, scale=1)
    p = normal("z_1", loc=0, scale=4)
    fwd = mlp_kernel("z_1", dim_hidden=4)
    rev = mlp_kernel("z_0", dim_hidden=4)

    optimizer = torch.optim.Adam(
        [dict(params=x.program.parameters()) for x in [p, q, fwd, rev]], lr=0.5
    )

    def hash_kernel_parameters():
        return [[thash(t) for t in kern.program.parameters()] for kern in [fwd, rev]]

    fwd_hashes_0, rev_hashes_0 = hash_kernel_parameters()

    assert all([len(list(prim.program.parameters())) == 0 for prim in [p, q]])

    q_ext = compose(fwd, q)
    p_ext = extend(p, rev)
    prop = propose(p=p_ext, q=q_ext, loss_fn=nvo_avo)

    with pyro.plate("sample", 7):
        out = prop()
        loss = out.nodes[_LOSS]["value"]

        loss.backward()
        optimizer.step()

    fwd_hashes_1, rev_hashes_1 = hash_kernel_parameters()

    assert all([l != r for l, r in zip(fwd_hashes_0, fwd_hashes_1)])
    assert all([l != r for l, r in zip(rev_hashes_0, rev_hashes_1)])


def vsmc(targets, forwards, reverses, loss_fn, resample=False, stl=False):
    q = targets[0]
    for ix, (fwd, rev, p) in enumerate(zip(forwards, reverses, targets[1:])):
        q = propose(
            p=extend(p, rev),
            q=compose(fwd, q),
            loss_fn=loss_fn,
        )
        if stl:
            q = augment_logweight(q, pre=stl_trick)
        if resample and ix < len(forwards) - 1:
            # NOTE: requires static knowledge that you plate once!
            q = combinator.resample(q, batch_dim=0, sample_dim=1)
    return q


def test_avo_1step(normal, mlp_kernel):
    pyro.set_rng_seed(7)
    q = normal("z_1", loc=1, scale=0.05)
    p = normal("z_2", loc=2, scale=0.05)
    fwd = mlp_kernel(p.program.name, dim_hidden=10)
    rev = mlp_kernel(q.program.name, dim_hidden=10)
    inf_step = vsmc(
        targets=[q, p], forwards=[fwd], reverses=[rev], loss_fn=nvo_avo, resample=False
    )

    optimizer = torch.optim.Adam(
        params=[dict(params=x.program.parameters()) for x in [fwd, rev]], lr=0.2
    )

    for i in range(100):
        optimizer.zero_grad()
        with pyro.plate("samples", 50):
            out = inf_step()
            loss = out.nodes[_LOSS]["value"]
            loss.backward()
            optimizer.step()

    with pyro.plate("samples", 1000):
        target_loc = p().nodes["z_2"]["value"].mean()
        forward_loc = compose(fwd, q)().nodes[_RETURN]["value"].mean()
        # pretty huge epsilon for a very short runway (100 steps).
        assert abs(target_loc - forward_loc) < 0.15


def test_avo_construction(normal, mlp_kernel, num_targets=3):
    pyro.set_rng_seed(7)
    targets = [normal(f"z_{i}", loc=i, scale=0.25) for i in range(1, num_targets + 1)]
    assert len(targets) > 2, "1 step accounted for in test_avo_1step"

    target_addrs = [t.program.name for t in targets]
    forward_addrs, reverse_addrs = target_addrs[1:], target_addrs[:-1]
    forwards, reverses = [
        [mlp_kernel(addr, dim_hidden=4) for addr in addrs]
        for addrs in [forward_addrs, reverse_addrs]
    ]
    assert len(forwards) == len(reverses) and len(forwards) == num_targets - 1
    return targets, forwards, reverses


def test_avo_2step_grads_dont_leak(normal, mlp_kernel):
    targets, forwards, reverses = test_avo_construction(
        normal, mlp_kernel, num_targets=3
    )

    q0, q1, q2 = targets
    f01, f12 = forwards
    r10, r21 = reverses

    @nested_objective
    def nvo_avo1(p_out, q_out, lw, lv, sample_dims=-1) -> Tensor:
        return (-lv).sum(dim=(sample_dims,), keepdim=False)

    q = propose(p=extend(q1, r10), q=compose(f01, q0), loss_fn=nvo_avo1)

    @nested_objective
    def nvo_avo2(p_out, q_out, lw, lv, sample_dims=-1) -> Tensor:
        loss = (-lv).sum(dim=(sample_dims,), keepdim=False)
        loss.backward()
        return loss

    q = propose(p=extend(q2, r21), q=compose(f12, q), loss_fn=nvo_avo2)

    optimizer = torch.optim.Adam(
        params=[dict(params=x.program.parameters()) for x in forwards + reverses],
        lr=0.2,
    )

    # TODO: remove thash, make this test more direct. As it stands, it is not clear that
    # this check is evaluating the right thing. Looking at grad is probably more precise
    #
    # Later asserts of `==` should just be torch.equal.
    first_params_pre = [
        thash(param)
        for first_kernel in [f01, r10]
        for param in first_kernel.program.parameters()
    ]
    second_params_pre = [
        thash(param)
        for second_kernel in [f12, r21]
        for param in second_kernel.program.parameters()
    ]

    out = q()
    optimizer.step()

    first_params_post = [
        thash(param)
        for first_kernel in [f01, r10]
        for param in first_kernel.program.parameters()
    ]
    second_params_post = [
        thash(param)
        for second_kernel in [f12, r21]
        for param in second_kernel.program.parameters()
    ]

    assert all(
        [pre == post for pre, post in zip(first_params_pre, first_params_post)]
    ), "no parameters change in first layer of nesting"
    assert all(
        [pre != post for pre, post in zip(second_params_pre, second_params_post)]
    ), "all parameters change in second layer of nesting"


@mark.skip("still being developed, but commiting this test as a checkpoint")
def test_avo_2step_grad_surgery_is_precise(normal, mlp_kernel):
    seq_length = 3
    targets, forwards, reverses = test_avo_construction(
        normal, mlp_kernel, num_targets=seq_length
    )

    q0, q1, q2 = targets
    f01, f12 = forwards
    r10, r21 = reverses

    def sitep(fn):
        return lambda kv: fn(kv[1])

    def verify_grads(p_out, q_out):
        # check proposal trace
        print("---------")
        for k, s in filter(sitep(is_sample_type), q_out.nodes.items()):
            g = s["value"].requires_grad
            print("q  ", k, g)

        sampled_sites = dict(filter(sitep(is_sample_type), p_out.nodes.items()))
        aux_samples = dict(filter(sitep(is_auxiliary), sampled_sites.items()))
        out_samples = dict(
            filter(sitep(lambda x: not is_auxiliary(x)), sampled_sites.items())
        )
        # check target trace
        for k, aux in aux_samples.items():
            g = aux["value"].requires_grad
            print("aux", k, g)

        for k, out in out_samples.items():
            g = out["value"].requires_grad
            print("out", k, g)
        for k in out_samples.keys():
            assert not q_out.nodes[k]["value"].requires_grad

    @nested_objective
    def nvo_avo1(p_out, q_out, lw, lv, sample_dims=-1) -> Tensor:
        # verify_grads(p_out, q_out)
        return (-lv).sum(dim=(sample_dims,), keepdim=False)

    q = propose(p=extend(q1, r10), q=compose(f01, q0), loss_fn=nvo_avo1)

    @nested_objective
    def nvo_avo2(p_out, q_out, lw, lv, sample_dims=-1) -> Tensor:
        verify_grads(p_out, q_out)
        loss = (-lv).sum(dim=(sample_dims,), keepdim=False)
        loss.backward()
        return loss

    q = propose(p=extend(q2, r21), q=compose(f12, q), loss_fn=nvo_avo2)

    optimizer = torch.optim.Adam(
        params=[dict(params=x.program.parameters()) for x in forwards + reverses],
        lr=0.2,
    )

    # remove thash
    first_params_pre = [
        thash(
            param
        )  # TODO: is this check evaluating the right thing? looking at grad is probably more precise
        for first_kernel in [f01, r10]
        for param in first_kernel.program.parameters()
    ]
    second_params_pre = [
        thash(param)
        for second_kernel in [f12, r21]
        for param in second_kernel.program.parameters()
    ]

    out = q()
    optimizer.step()

    first_params_post = [
        thash(param)
        for first_kernel in [f01, r10]
        for param in first_kernel.program.parameters()
    ]
    second_params_post = [
        thash(param)
        for second_kernel in [f12, r21]
        for param in second_kernel.program.parameters()
    ]
    # # TODO: is == torch.equal?
    # assert all(
    #     [pre == post for pre, post in zip(first_params_pre, first_params_post)]
    # ), "no parameters change in first layer of nesting"
    # assert all(
    #     [pre != post for pre, post in zip(second_params_pre, second_params_post)]
    # ), "all parameters change in second layer of nesting"


def test_avo_2step_empirical(normal, mlp_kernel):
    num_targets = 3
    targets, forwards, reverses = test_avo_construction(
        normal, mlp_kernel, num_targets=num_targets
    )

    infer = vsmc(
        targets=targets,
        forwards=forwards,
        reverses=reverses,
        loss_fn=nvo_avo,
        resample=False,
    )
    optimizer = torch.optim.Adam(
        params=[dict(params=x.program.parameters()) for x in forwards + reverses],
        lr=0.2,
    )

    for _ in range(250):
        optimizer.zero_grad()
        out = infer()
        loss = out.nodes[_LOSS]["value"]
        loss.backward()
        optimizer.step()

    chain = reduce(lambda q, fwd: compose(fwd, q), forwards, targets[0])
    target = targets[-1]

    with pyro.plate("samples", 1000):
        target_loc = target().nodes[f"z_{num_targets}"]["value"].mean().detach()
        forward_loc = chain().nodes[_RETURN]["value"].mean().detach()
        assert abs(num_targets - target_loc) < 0.05, "target location is miscalculated"
        assert (
            abs(target_loc - forward_loc) < 0.15
        ), "kernels did not learn target density"


def test_avo_4step_empirical(normal, mlp_kernel):
    pyro.set_rng_seed(7)
    num_targets = 5
    targets, forwards, reverses = test_avo_construction(
        normal, mlp_kernel, num_targets=num_targets
    )
    infer = vsmc(
        targets=targets,
        forwards=forwards,
        reverses=reverses,
        loss_fn=nvo_avo,
        resample=False,
    )

    optimizer = torch.optim.Adam(
        params=[dict(params=x.program.parameters()) for x in forwards + reverses],
        lr=0.01,
    )

    for _ in range(500):
        optimizer.zero_grad()
        out = infer()
        loss = out.nodes[_LOSS]["value"]
        loss.backward()
        optimizer.step()

    chain = reduce(lambda q, fwd: compose(fwd, q), forwards, targets[0])
    target = targets[-1]

    with pyro.plate("samples", 1000):
        target_loc = target().nodes[f"z_{num_targets}"]["value"].mean().detach()
        forward_loc = chain().nodes[_RETURN]["value"].mean().detach()
        assert abs(num_targets - target_loc) < 0.05, "target location is miscalculated"
        assert (
            abs(target_loc - forward_loc) < 0.15
        ), "kernels did not learn target density"


def test_avo_4step_empirical_with_stl(normal, mlp_kernel):
    pyro.set_rng_seed(7)
    num_targets = 5
    targets, forwards, reverses = test_avo_construction(
        normal, mlp_kernel, num_targets=num_targets
    )

    infer = vsmc(
        targets=targets,
        forwards=forwards,
        reverses=reverses,
        loss_fn=nvo_avo,
        stl=True,
    )

    optimizer = torch.optim.Adam(
        params=[dict(params=x.program.parameters()) for x in forwards + reverses],
        lr=0.001,
    )
    from tqdm import trange

    losses = []
    with trange(2000) as bar:
        for i in bar:
            optimizer.zero_grad()
            out = infer()
            loss = out.nodes[_LOSS]["value"]
            loss.backward()
            optimizer.step()

            # REPORTING
            losses.append(loss.detach().cpu().mean().item())
            if len(losses) > 100:
                losses.pop(0)
                if i % 10 == 0:
                    loss_scalar = sum(losses) / len(losses)
                    bar.set_postfix_str(
                        "loss={}{:.3f}".format(
                            "" if loss_scalar < 0 else " ", loss_scalar
                        )
                    )

    chain = reduce(lambda q, fwd: compose(fwd, q), forwards, targets[0])
    target = targets[-1]

    with pyro.plate("samples", 1000):
        target_loc = target().nodes[f"z_{num_targets}"]["value"].mean().detach()
        forward_loc = chain().nodes[_RETURN]["value"].mean().detach()
        assert abs(num_targets - target_loc) < 0.05, "target location is miscalculated"
        assert (
            abs(target_loc - forward_loc) < 0.15
        ), "kernels did not learn target density"


def test_avo_4step_empirical_with_resampling(normal, mlp_kernel):
    pyro.set_rng_seed(7)
    num_targets = 5
    targets, forwards, reverses = test_avo_construction(
        normal, mlp_kernel, num_targets=num_targets
    )
    infer = vsmc(
        targets=targets,
        forwards=forwards,
        reverses=reverses,
        loss_fn=nvo_avo,
        resample=True,
    )

    optimizer = torch.optim.Adam(
        params=[dict(params=x.program.parameters()) for x in forwards + reverses],
        lr=0.01,
    )

    for _ in range(500):
        optimizer.zero_grad()
        with pyro.plate("samples", 50):
            out = infer()
            loss = out.nodes[_LOSS]["value"]
            loss.backward()
        optimizer.step()

    chain = reduce(lambda q, fwd: compose(fwd, q), forwards, targets[0])
    target = targets[-1]

    with pyro.plate("samples", 1000):
        target_loc = target().nodes[f"z_{num_targets}"]["value"].mean().detach()
        forward_loc = chain().nodes[_RETURN]["value"].mean().detach()
        assert abs(num_targets - target_loc) < 0.05, "target location is miscalculated"
        assert (
            abs(target_loc - forward_loc) < 0.015
        ), "kernels did not learn target density"
