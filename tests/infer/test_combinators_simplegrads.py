# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools
from operator import getitem, itemgetter
from pyro.poutine.handlers import _make_handler
from pyro.poutine.messenger import Messenger
from typing import Any, Callable, Optional, Union, Optional

import pytest
import torch
from torch import Tensor, tensor
import torch.nn as nn
from pytest import mark, fixture

import pyro
import pyro.distributions as dist
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
)
from pyro.poutine import Trace, replay
from pyro.nn.module import PyroModule

#@pytest.fixture(scope="session", autouse=True)
#def mlp_kernel():
#    def model():
#        z_3 = pyro.sample("z_3", dist.Normal(tensor_of(3), tensor_of(3)))
#
#        pyro.sample("x_3", dist.Normal(tensor_of(3), tensor_of(3)), obs=z_3)
#
#        return None
#
#    yield model
#
#class MLPKernel(nn.PyroModule):
#    @PyroSample
#    def x(self):
#        return Normal(0, 1)       # independent
#
#    @PyroSample
#    def y(self):
#        return Normal(self.x, 1)  # dependent
#
#    def forward(self):
#        return self.y             # accessed like a @property
#
#class MLPKernel(Kernel):
#    def __init__(self, dim_hidden, ext_name):
#        super().__init__()
#        self.ext_name = ext_name
#        self.net = nn.Sequential(
#            nn.Linear(1, dim_hidden),
#            nn.Sigmoid(),
#            nn.Linear(dim_hidden, dim_hidden),
#            nn.Sigmoid(),
#            nn.Linear(dim_hidden, 1),
#        )
#
#    def apply_kernel(self, trace, cond_trace, obs):
#        return trace.normal(
#            loc=self.net(obs.detach()),
#            scale=torch.ones(1, device=obs.device),
#            value=None
#            if self.ext_name not in cond_trace
#            else cond_trace[self.ext_name].value,
#            name=self.ext_name,
#        )

class MLPKernel(PyroModule):
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
        return pyro.sample("g", dist.Normal(loc=self.net(obs.detach()), scale=torch.one(1)))

def test_forward():
    g = lambda: pyro.sample("g", dist.Normal(loc=torch.zeros(1), scale=torch.ones(1)))
    fwd = MLPKernel("fwd", dim_hidden=4,)

    ext = compose(fwd, g)
    ext()
    breakpoint();
    print()


#def test_forward_forward(seed):
#    g0 = Normal(loc=0, scale=1, name="g0")
#    f01 = MLPKernel(dim_hidden=4, ext_name="g1").to(autodevice())
#    f12 = MLPKernel(dim_hidden=4, ext_name="g2").to(autodevice())
#
#    ext = Forward(f12, Forward(f01, g0))
#    ext()
#
#
#def test_reverse(seed):
#    g = Normal(loc=0, scale=1, name="g")
#    rev = MLPKernel(dim_hidden=4, ext_name="rev").to(autodevice())
#
#    ext = Reverse(g, rev)
#    ext()
#
#    for k in ext._cache.program.trace.keys():
#        assert torch.equal(
#            ext._cache.program.trace[k].value, ext._cache.kernel.trace[k].value
#        )
#
#
#def test_propose_values(seed):
#    q = Normal(loc=4, scale=1, name="z_0")
#    p = Normal(loc=0, scale=4, name="z_1")
#    fwd = MLPKernel(dim_hidden=4, ext_name="z_1").to(autodevice())
#    rev = MLPKernel(dim_hidden=4, ext_name="z_0").to(autodevice())
#    optimizer = torch.optim.Adam(
#        [dict(params=x.parameters()) for x in [p, q, fwd, rev]], lr=0.5
#    )
#    assert len(list(p.parameters())) == 0
#    assert len(list(q.parameters())) == 0
#    fwd_hashes_0 = [thash(t) for t in fwd.parameters()]
#    rev_hashes_0 = [thash(t) for t in rev.parameters()]
#
#    q_ext = Forward(fwd, q)
#    p_ext = Reverse(p, rev)
#    extend = Propose(target=p_ext, proposal=q_ext)
#
#    log_log_prob = extend().log_prob
#
#    assert isinstance(log_log_prob, Tensor)
#
#    proposal_cache = extend.proposal._cache
#    target_cache = extend.target._cache
#
#    for k in ["z_0", "z_1"]:
#        assert torch.equal(
#            proposal_cache.kernel.trace[k].value, target_cache.kernel.trace[k].value
#        )
#
#    loss = nvo_avo(log_log_prob, sample_dims=0).mean()
#    loss.backward()
#
#    optimizer.step()
#    fwd_hashes_1 = [thash(t) for t in fwd.parameters()]
#    rev_hashes_1 = [thash(t) for t in rev.parameters()]
#
#    assert any([l != r for l, r in zip(fwd_hashes_0, fwd_hashes_1)])
#    assert any([l != r for l, r in zip(rev_hashes_0, rev_hashes_1)])
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
#
#def test_1step_avo(seed, is_smoketest):
#    """ The VAE test. At one step no need for any detaches. """
#
#    target_params, proposal_params = all_params = [Params(4, 1), Params(1, 4)]
#    target, proposal = [Normal(*p, name=f"z_{p.loc}") for p in all_params]
#    # fwd, rev = [MLPKernel(dim_hidden=4, ext_name=f'z_{ext_mean}') for ext_mean in [4, 1]]
#    fwd = NormalLinearKernel(ext_from=f"z_1", ext_to="z_4").to(autodevice())
#    rev = NormalLinearKernel(ext_from=f"z_4", ext_to="z_1").to(autodevice())
#
#    optimizer = torch.optim.Adam(
#        [dict(params=x.parameters()) for x in [proposal, target, fwd, rev]], lr=0.1
#    )
#
#    num_steps = 1 if is_smoketest else 100
#    loss_ct, loss_sum, loss_avgs, loss_all = 0, 0.0, [], []
#
#    with trange(num_steps) as bar:
#        for i in bar:
#            optimizer.zero_grad()
#            q_ext = Forward(fwd, proposal)
#            p_ext = Reverse(target, rev)
#            extend = Propose(target=p_ext, proposal=q_ext)
#
#            log_log_prob = extend(sample_shape=(5, 1), sample_dims=0).log_prob
#
#            # proposal.clear_observations() # FIXME: this can be automated, but it would be nice to have more infrastructure around observations
#            loss = nvo_avo(log_log_prob).mean()
#
#            loss.backward()
#
#            optimizer.step()
#
#            # REPORTING
#            loss_ct += 1
#            loss_scalar = loss.detach().cpu().mean().item()
#            loss_sum += loss_scalar
#            loss_all.append(loss_scalar)
#            if num_steps <= 100:
#                loss_avgs.append(loss_scalar)
#            if i % 10 == 0:
#                loss_avg = loss_sum / loss_ct
#                loss_template = "loss={}{:.4f}".format(
#                    "" if loss_avg < 0 else " ", loss_avg
#                )
#                bar.set_postfix_str(loss_template)
#                loss_ct, loss_sum = 0, 0.0
#                if num_steps > 100:
#                    loss_avgs.append(loss_avg)
#    with torch.no_grad():
#        assert_empirical_marginal_mean_std(
#            lambda: Forward(fwd, proposal)().output,
#            target_params,
#            Tolerance(loc=0.15, scale=0.15),
#            is_smoketest,
#            5 if is_smoketest else 400,
#        )
#
#
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
