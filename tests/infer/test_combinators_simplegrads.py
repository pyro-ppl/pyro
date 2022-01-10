# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from pytest import mark, fixture
from torch import Tensor, tensor
from typing import Any, Callable, Optional, Union, Optional
from tqdm import trange

import pyro
import pyro.distributions as dist
import pyro.optim.pytorch_optimizers as optim
from pyro import poutine
from pyro.infer.combinators import (
    Node,
    Trace,
    addr_filter,
    compose,
    extend,
    get_marginal,
    is_auxiliary,
    membership_filter,
    not_auxiliary,
    primitive,
    propose,
    with_substitution,
    _LOGWEIGHT, _LOSS, _RETURN
)
from pyro.nn.module import PyroModule
from pyro.poutine import Trace, replay
from pyro.poutine.handlers import _make_handler
from pyro.poutine.messenger import Messenger



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



class MLPKernel(PyroModule):
    """ simple MLP kernel that expects an observation with a sample dimension """
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
        return pyro.sample(self.name, dist.Normal(loc=self.net(obs.T.detach()).T, scale=1.))


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
        return pyro.sample(self.name, dist.Normal(loc=torch.ones(1,1)*self.loc, scale=torch.ones(1,1)*self.scale))


@pytest.fixture(scope="session", autouse=True)
def normal():
    def model(name:str, loc:int=0, scale:int=1):
        return primitive(Normal(name, loc, scale))
    yield model


def nvo_avo(p_out, q_out, lw, lv, sample_dims=-1) -> Tensor:
    def compute(lv):
        # internal function just to note that we only need lv for avo
        nw = F.softmax(torch.zeros_like(lv), dim=sample_dims)
        loss = (nw * (-lv)).sum(dim=(sample_dims,), keepdim=False)
        return loss
    return compute(lv)


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
        loss = out.nodes[_LOSS]['value']

        loss.backward()
        optimizer.step()

    fwd_hashes_1, rev_hashes_1 = hash_kernel_parameters()

    assert all([l != r for l, r in zip(fwd_hashes_0, fwd_hashes_1)])
    assert all([l != r for l, r in zip(rev_hashes_0, rev_hashes_1)])


def vsmc(targets, forwards, reverses, loss_fn, resample=False):
    q = targets[0]
    for ix, (fwd, rev, p) in enumerate(zip(forwards, reverses, targets[1:])):
        q = propose(
            p=extend(p, rev),
            q=compose(fwd, q),
            loss_fn=loss_fn,
            # FIXME: reintroduce
            # transf_q_trace=None if loss_fn.__name__ == "nvo_avo" else stl_trace,
        )
        # FIXME: reintroduce
        # if resample and k < len(forwards) - 1:
        #     q = Resample(q)
    return q


def test_avo_1step(normal, mlp_kernel):
    pyro.set_rng_seed(7)

    q = normal("z_1", loc=1, scale=0.05)
    p = normal("z_2", loc=4, scale=0.05)
    fwd = mlp_kernel("fwd_12", dim_hidden=10)
    rev = mlp_kernel("rev_21", dim_hidden=10)
    inf_step = vsmc(targets=[q, p], forwards=[fwd], reverses=[rev], loss_fn=nvo_avo, resample=False)

    optimizer = torch.optim.Adam(
        params=[dict(params=x.program.parameters()) for x in [fwd, rev]], lr=0.2
    )

    with trange(1000) as bar:
        for i in bar:
            optimizer.zero_grad()
            out = inf_step()
            loss = out.nodes[_LOSS]['value']
            loss.backward()

            # REPORTING
            if i % 100 == 0:
                loss_scalar = loss.detach().cpu().mean().item()
                bar.set_postfix_str("loss={}{:.4f}".format(
                    "" if loss_scalar < 0 else " ", loss.detach().cpu().mean().item()
                ))

    with pyro.plate("samples", 1000):
        target_loc = p().nodes['z_2']['value'].mean()
        forward_loc = compose(fwd, q)().nodes[_RETURN]['value'].mean()
        # pretty huge epsilon for a very short runway (100 steps).
        assert abs(target_loc - forward_loc) < 0.15


def test_avo_4step(normal, mlp_kernel):
    pyro.set_rng_seed(7)
    num_targets = 4
    targets = [normal(f"z_{i}", loc=i, scale=0.25) for i in range(1, num_targets+1)]
    assert len(targets) > 2, "1 step accounted for in test_avo_1step"

    forwards = [mlp_kernel(f"fwd_{i}{i+1}", dim_hidden=10) for i in range(1, num_targets,  1)]
    reverses = [mlp_kernel(f"rev_{i}{i-1}", dim_hidden=10) for i in range(num_targets, 1, -1)]

    inf_step = vsmc(targets=targets, forwards=forwards, reverses=reverses, loss_fn=nvo_avo, resample=False)

    optimizer = torch.optim.Adam(
        params=[dict(params=x.program.parameters()) for x in forwards+reverses], lr=0.2
    )

    with trange(1000) as bar:
        for i in bar:
            optimizer.zero_grad()
            out = inf_step()
            loss = out.nodes[_LOSS]['value']
            loss.backward()

            # REPORTING
            if i % 100 == 0:
                loss_scalar = loss.detach().cpu().mean().item()
                bar.set_postfix_str("loss={}{:.4f}".format(
                    "" if loss_scalar < 0 else " ", loss.detach().cpu().mean().item()
                ))

    chain = reduce(lambda q, fwd: compose(fwd, q), forwards, targets[0])
    target = targets[-1]
    chain = compose(forwards[0], targets[0])

    with pyro.plate("samples", 1000):
        target_loc = target().nodes[f'z_{num_targets}']['value'].mean().detach()
        forward_loc = chain().nodes[_RETURN]['value'].mean().detach()
        assert abs(num_targets - target_loc) < 0.05, "target location is miscalculated"
        assert abs(target_loc - forward_loc) < 0.15, "kernels did not learn target density"




#def test_2step_avo(seed, use_fast, is_smoketest):
#    """
#    2-step NVI (NVI-sequential): 4 intermediate densities (target and proposal always fixed).
#
#    With four steps, you'll need to detach whenever you compute a normalizing constant in all the intermediate steps.
#    """
#    g1, g2, g3 = targets = [Normal(loc=i, scale=1, name=f"z_{i}") for i in range(1, 4)]
#    f12, f23 = forwards = [
#        NormalLinearKernel(ext_from=f"z_{i}", ext_to=f"z_{i+1}").to(autodevice())
#        for i in range(1, 3)
#    ]
#    r21, r32 = reverses = [
#        NormalLinearKernel(ext_from=f"z_{i+1}", ext_to=f"z_{i}").to(autodevice())
#        for i in range(1, 3)
#    ]
#
#    optimizer = torch.optim.Adam(
#        [dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]],
#        lr=0.1 if use_fast else 1e-2,
#    )
#
#    num_steps = 5 if is_smoketest else (100 if use_fast else 400)
#    loss_ct, loss_sum, loss_all = 0, 0.0, []
#    lvs_all = []
#    sample_shape = (100, 1)
#
#    with trange(num_steps) as bar:
#        for i in bar:
#            q0 = targets[0]
#            p_prv_tr, _, out0 = q0(sample_shape=sample_shape)
#
#            loss = torch.zeros([1], **kw_autodevice())
#
#            lvs = []
#            for fwd, rev, q, p in zip(forwards, reverses, targets[:-1], targets[1:]):
#                # q.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
#                q_ext = Forward(fwd, q)
#                p_ext = Reverse(p, rev)
#                extend_argument = Propose(target=p_ext, proposal=q_ext)
#                # state, lv = extend_argument(sample_shape=sample_shape) # TODO
#                state = extend_argument(sample_shape=sample_shape, sample_dims=0)
#                lv = state.log_prob
#
#                lvs.append(lv)
#
#                p_prv_tr = state.trace
#                loss += nvo_avo(lv, sample_dims=0).mean()
#
#            lvs_ten = torch.stack(lvs, dim=0)
#            lvs_all.append(lvs_ten)
#
#            loss.backward()
#
#            optimizer.step()
#            optimizer.zero_grad()
#
#            # REPORTING
#            loss_ct += 1
#            loss_scalar = loss.detach().cpu().mean().item()
#            loss_sum += loss_scalar
#            loss_all.append(loss_scalar)
#            if i % 10 == 0:
#                loss_avg = loss_sum / loss_ct
#                loss_template = "loss={}{:.4f}".format(
#                    "" if loss_avg < 0 else " ", loss_avg
#                )
#                bar.set_postfix_str(loss_template)
#                loss_ct, loss_sum = 0, 0.0
#
#    with torch.no_grad():
#        # report.sparkline(loss_avgs)
#        lvs = torch.stack(lvs_all, dim=0)
#        lws = torch.cumsum(lvs, dim=1)
#        ess = effective_sample_size(lws)
#        # import matplotlib.pyplot as plt
#        # plt.plot(ess)
#        # plt.savefig("fig.png")
#
#        # This is the analytic marginal for the forward kernel
#        out12 = propagate(
#            N=g1.as_dist(as_multivariate=True),
#            F=f12.weight(),
#            t=f12.bias(),
#            B=torch.eye(1, **kw_autodevice()),
#            marginalize=True,
#        )
#        print(out12.loc)
#        out23 = propagate(
#            N=g2.as_dist(as_multivariate=True),
#            F=f23.weight(),
#            t=f23.bias(),
#            B=torch.eye(1, **kw_autodevice()),
#            marginalize=True,
#        )
#        print(out23.loc)
#
#        tol = Tolerance(loc=0.15, scale=0.15)
#        assert is_smoketest or abs((out12.loc - g2.dist.loc).item()) < tol.loc
#        assert is_smoketest or abs((out23.loc - g3.dist.loc).item()) < tol.loc
#
#        tr, _, out = g1(sample_shape=(200, 1))
#        assert is_smoketest or abs(out.mean().item() - 1) < tol.loc
#        tr, _, out = f12(tr, out)
#        assert is_smoketest or abs(out.mean().item() - 2) < tol.loc
#
#        pre2 = Forward(f12, g1)
#        tr, _, out = pre2(sample_shape=(200, 1))
#        assert is_smoketest or abs(out.mean().item() - 2) < tol.loc
#
#        tr, _, out = g2(sample_shape=(200, 1))
#        assert is_smoketest or abs(out.mean().item() - 2) < tol.loc
#        tr, _, out = f23(tr, out)
#        assert is_smoketest or abs(out.mean().item() - 3) < tol.loc
#
#        pre3 = Forward(f23, g2)
#        tr, _, out = pre3(sample_shape=(200, 1))
#        assert is_smoketest or abs(out.mean().item() - 3) < tol.loc
#
#        predict_g1_to_g2 = lambda: pre2().output
#        predict_g2_to_g3 = lambda: pre3().output
#        predict_g1_to_g3 = lambda: Forward(f23, Forward(f12, g1))().output
#
#        assert_empirical_marginal_mean_std(
#            predict_g1_to_g2,
#            Params(loc=2, scale=1),
#            tol,
#            is_smoketest,
#            5 if is_smoketest else 400,
#        )
#        assert_empirical_marginal_mean_std(
#            predict_g2_to_g3,
#            Params(loc=3, scale=1),
#            tol,
#            is_smoketest,
#            5 if is_smoketest else 400,
#        )
#        assert_empirical_marginal_mean_std(
#            predict_g1_to_g3,
#            Params(loc=3, scale=1),
#            tol,
#            is_smoketest,
#            5 if is_smoketest else 400,
#        )
#
#
#def test_4step_avo(seed, use_fast, is_smoketest):
#    """
#    4-step NVI-sequential: 8 intermediate densities
#    """
#    g1, g2, g3, g4, g5 = targets = [
#        Normal(loc=i, scale=1, name=f"z_{i}") for i in range(1, 6)
#    ]
#    f12, f23, f34, f45 = forwards = [
#        NormalLinearKernel(ext_from=f"z_{i}", ext_to=f"z_{i+1}").to(autodevice())
#        for i in range(1, 5)
#    ]
#    r21, r32, r43, r54 = reverses = [
#        NormalLinearKernel(ext_from=f"z_{i+1}", ext_to=f"z_{i}").to(autodevice())
#        for i in range(1, 5)
#    ]
#    assert r21.ext_to == "z_1"
#    assert f12.ext_to == "z_2"
#    assert r54.ext_to == "z_4"
#    assert f45.ext_to == "z_5"
#
#    optimizer = torch.optim.Adam(
#        [dict(params=x.parameters()) for x in [*forwards, *reverses, *targets]],
#        lr=0.4 if use_fast else 1e-2,
#    )
#
#    num_steps = 5 if is_smoketest else (100 if use_fast else 1000)
#    sample_shape = (100, 1)
#    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []
#
#    with trange(num_steps) as bar:
#        for i in bar:
#            q0 = targets[0]
#            p_prv_tr, _, out0 = q0(sample_shape=sample_shape)
#            loss = torch.zeros(1, **kw_autodevice())
#
#            lvs = []
#            for fwd, rev, q, p in zip(forwards, reverses, targets[:-1], targets[1:]):
#                # q.with_observations(trace_utils.copytrace(p_prv_tr, detach=p_prv_tr.keys()))
#                q_ext = Forward(fwd, q)
#                p_ext = Reverse(p, rev)
#                extend_argument = Propose(target=p_ext, proposal=q_ext)
#                # state, lv = extend_argument(sample_shape=sample_shape) # TODO
#                state = extend_argument(sample_shape=sample_shape, sample_dims=0)
#                # q.clear_observations()
#                # p.clear_observations()
#                lv = state.log_prob
#
#                lvs.append(lv)
#
#                p_prv_tr = state.trace
#                loss += nvo_avo(lv, sample_dims=0).mean()
#
#            loss.backward()
#
#            optimizer.step()
#            optimizer.zero_grad()
#
#            # REPORTING
#            loss_ct += 1
#            loss_scalar = loss.detach().cpu().mean().item()
#            loss_sum += loss_scalar
#            loss_all.append(loss_scalar)
#            if i % 10 == 0:
#                loss_avg = loss_sum / loss_ct
#                loss_template = "loss={}{:.4f}".format(
#                    "" if loss_avg < 0 else " ", loss_avg
#                )
#                bar.set_postfix_str(loss_template)
#                loss_ct, loss_sum = 0, 0.0
#
#    with torch.no_grad():
#        tol = Tolerance(loc=0.15, scale=0.15)
#
#        out12 = propagate(
#            N=g1.as_dist(as_multivariate=True),
#            F=f12.weight(),
#            t=f12.bias(),
#            B=torch.eye(1),
#            marginalize=True,
#        )
#        out23 = propagate(
#            N=g2.as_dist(as_multivariate=True),
#            F=f23.weight(),
#            t=f23.bias(),
#            B=torch.eye(1),
#            marginalize=True,
#        )
#        out34 = propagate(
#            N=g3.as_dist(as_multivariate=True),
#            F=f34.weight(),
#            t=f34.bias(),
#            B=torch.eye(1),
#            marginalize=True,
#        )
#        out45 = propagate(
#            N=g4.as_dist(as_multivariate=True),
#            F=f45.weight(),
#            t=f45.bias(),
#            B=torch.eye(1),
#            marginalize=True,
#        )
#        for (analytic, target_loc) in zip([out12, out23, out34, out45], range(2, 6)):
#            assert is_smoketest or (target_loc - analytic.loc.item()) < tol.loc
#
#        predict_g2_chain = lambda: Forward(f12, g1)().output
#        predict_g3_chain = lambda: Forward(f23, Forward(f12, g1))().output
#        predict_g4_chain = lambda: Forward(f34, Forward(f23, Forward(f12, g1)))().output
#        predict_g5_chain = lambda: Forward(
#            f45, Forward(f34, Forward(f23, Forward(f12, g1)))
#        )().output
#
#        if not use_fast:
#            eval_mean_std(
#                predict_g2_chain,
#                Params(loc=2, scale=1),
#                tol,
#                is_smoketest,
#                5 if is_smoketest else 400,
#            )
#            eval_mean_std(
#                predict_g3_chain,
#                Params(loc=3, scale=1),
#                tol,
#                is_smoketest,
#                5 if is_smoketest else 400,
#            )
#            eval_mean_std(
#                predict_g4_chain,
#                Params(loc=4, scale=1),
#                tol,
#                is_smoketest,
#                5 if is_smoketest else 400,
#            )
#            eval_mean_std(
#                predict_g5_chain,
#                Params(loc=5, scale=1),
#                tol,
#                is_smoketest,
#                5 if is_smoketest else 400,
#            )
#
#        predict_g2 = lambda: Forward(f12, g1)().output
#        predict_g3 = lambda: Forward(f23, g2)().output
#        predict_g4 = lambda: Forward(f34, g3)().output
#        predict_g5 = lambda: Forward(f45, g4)().output
#
#        eval_mean_std(
#            predict_g2,
#            Params(loc=2, scale=1),
#            tol,
#            is_smoketest,
#            5 if is_smoketest else 400,
#        )
#        eval_mean_std(
#            predict_g3,
#            Params(loc=3, scale=1),
#            tol,
#            is_smoketest,
#            5 if is_smoketest else 400,
#        )
#        eval_mean_std(
#            predict_g4,
#            Params(loc=4, scale=1),
#            tol,
#            is_smoketest,
#            5 if is_smoketest else 400,
#        )
#        eval_mean_std(
#            predict_g5,
#            Params(loc=5, scale=1),
#            tol,
#            is_smoketest,
#            5 if is_smoketest else 400,
#        )
#


# OLD SANITY CHECKS (IF NEEDED)
# -----------------------------
# def test_forward(mlp_kernel, normal):
#     seed(7)
#     g = normal("g", loc=0, scale=1)
#     fwd = mlp_kernel("fwd", dim_hidden=4,)
#     ext = compose(fwd, g)
#     ext()
#
#
# def test_forward_forward(mlp_kernel, normal):
#     seed(7)
#     g0 = normal("g0", loc=0, scale=1)
#     f01 = mlp_kernel("g1", dim_hidden=4)
#     f12 = mlp_kernel("g2", dim_hidden=4)
#
#     ext = compose(f12, compose(f01, g0))
#     ext()
#
#     #for k in ext._cache.program.trace.keys():
#     #    assert torch.equal(
#     #        ext._cache.program.trace[k].value, ext._cache.kernel.trace[k].value
#     #    )
#
#
#
# def test_reverse(mlp_kernel, normal):
#     g = normal("g", loc=0, scale=1)
#     rev = mlp_kernel("rev", dim_hidden=4)
#
#     ext = extend(g, rev)
#     ext()
#
#     #for k in ext._cache.program.trace.keys():
#     #    assert torch.equal(
#     #        ext._cache.program.trace[k].value, ext._cache.kernel.trace[k].value
#     #    )
#
#


#def test_propose_gradients(seed):
#    q = Normal(loc=4, scale=1, name="z_0", **kw_autodevice())
#    p = Normal(loc=0, scale=4, name="z_1", **kw_autodevice())
#    fwd = MLPKernel(dim_hidden=4, ext_name="z_1").to(autodevice())
#    rev = MLPKernel(dim_hidden=4, ext_name="z_0").to(autodevice())
#    optimizer = torch.optim.Adam(
#        [dict(params=x.parameters()) for x in [p, q, fwd, rev]], lr=0.5
#    )
#
#    tr0 = q().trace
#    q_ext = Forward(fwd, Condition(q, tr0, requires_grad=RequiresGrad.NO))
#    p_ext = Reverse(p, rev)
#    extend = Propose(target=p_ext, proposal=q_ext)
#
#    state = extend()
#    log_log_prob = state.log_prob
#
#    proposal_cache = extend.proposal._cache
#    target_cache = extend.target._cache
#
#    for k in ["z_0", "z_1"]:
#        assert torch.equal(
#            proposal_cache.kernel.trace[k].value, target_cache.kernel.trace[k].value
#        )
#
#    for k, prg in [
#        ("z_1", target_cache.kernel),
#        ("z_0", target_cache.kernel),
#        ("z_1", proposal_cache.kernel),
#    ]:
#        assert (
#            k == k and prg is prg and prg.trace[k].value.requires_grad
#        )  # k==k for debugging the assert
#
#    assert not proposal_cache.kernel.trace["z_0"].value.requires_grad
#
