#!/usr/bin/env python3
import matplotlib.pyplot as plt
import math
import numpy
import os
import pyro
import pyro.distributions as dist
from pyro.distributions.distribution import Distribution
import pyro.infer.combinators as combinator
import pyro.optim.pytorch_optimizers as optim
import sys
import time
import torch
import torch.nn.functional as F
from pytest import mark


from pyro.distributions import TorchDistribution
from pyro.nn import PyroModule, PyroParam
from pyro.infer.combinators import (
    compose,
    extend,
    primitive,
    propose,
    with_substitution,
    stl_trick,
    augment_logweight,
    nested_objective,
    _LOSS,
    _RETURN,
    is_sample_type,
    is_auxiliary,
)

from typing import Tuple
from torch import Tensor, nn
from functools import reduce
from scipy.interpolate import interpn
from matplotlib import cm

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import shutil

# ============================================================================================================
#                        plotting definitions
# ============================================================================================================
def plot_sample(
    ax, x, y=None, sort=True, bins=100, range=None, weight_cm=False, **kwargs
):
    ax.tick_params(
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )

    if y is None:
        assert x.shape[-1] == 2
        x, y = x[:, 0], x[:, 1]

    mz, x_e, y_e = numpy.histogram2d(
        x.numpy(), y.numpy(), bins=bins, density=True, range=range
    )
    if weight_cm:
        z = interpn(
            (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
            mz,
            numpy.vstack([x, y]).T,
            method="splinef2d",
            bounds_error=False,
        )
        z[numpy.where(numpy.isnan(z))] = 0.0  # To be sure to plot all data
        # Sort the points by density, so that the densest points are plotted last
        if sort:
            idx = z.argsort()
            x, y, z = x[idx], y[idx], z[idx]
        cmap = cm.get_cmap("viridis")
        ax.set_facecolor(cmap(0.0))
        ax.scatter(x, y, c=z, cmap=cmap, **kwargs)
        return mz
    else:
        ax.imshow(mz)


def plot_sequence(
    labels_and_Xs, dpi=100, fontsize=18, path=None, *plot_args, **plot_kwargs
):
    K = len(labels_and_Xs)
    fig = plt.figure(figsize=(3 * K, 3 * 1), dpi=dpi)
    for k, (label, X) in zip(range(K), labels_and_Xs):
        ax = fig.add_subplot(1, K, k + 1)
        ax.set_xlabel(label, fontsize=fontsize)
        plot_sample(ax, X, *plot_args, **plot_kwargs)
    if path is not None:
        plt.savefig(path)
    return fig


# ============================================================================================================
#                        Model definitions
# ============================================================================================================
class GMM(TorchDistribution):
    arg_constraints = {}

    def __init__(self, locs, covs, batch_shape=torch.Size()):
        super().__init__()
        assert len(locs.shape) == 2 and len(covs.shape) == 3
        assert locs.shape[0] == covs.shape[0]
        assert covs.shape[1] == covs.shape[2] and covs.shape[1] == locs.shape[1]

        self.K = locs.shape[0]
        self.locs = locs
        self.covs = covs
        self.components = [
            dist.MultivariateNormal(loc=locs[k], covariance_matrix=covs[k])
            for k in range(self.K)
        ]
        self.cluster_dist = dist.Categorical(torch.ones(self.K))
        self._batch_shape = batch_shape
        self._event_shape = torch.Size([locs.shape[1]])
        self.has_rsample = False

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape

    def sample(self, sample_shape=torch.Size([])):
        locs, covs, K = self.locs, self.covs, self.K
        zs = self.cluster_dist.sample(sample_shape)
        xs = []
        values, _ = torch.sort(zs)

        for k in range(self.K):
            n_k = (values == k).sum()
            if n_k > 0:
                x_k = self.components[k].sample(sample_shape=(n_k, *zs.shape[1:-1]))
                xs.append(x_k)
        xs = torch.cat(xs)
        ix = torch.randperm(xs.shape[0])
        return xs[ix].view(self.event_shape if len(zs.shape) == 0 else xs.shape)

    def log_prob(self, value):
        lds = []
        for i, comp in enumerate(self.components):
            ld_i = comp.log_prob(value)
            lds.append(ld_i)
        lds_ = torch.stack(lds, dim=0)
        ld = torch.logsumexp(lds_, dim=0)
        return ld


class RingGMM(GMM):
    def __init__(self, radius=10, scale=0.5, K=8):
        alpha = 2 * math.pi / K
        angles = alpha * torch.arange(K).float()
        x = radius * torch.sin(angles)
        y = radius * torch.cos(angles)
        locs = torch.stack((x, y), dim=0).T
        covs = torch.stack([torch.eye(2) * scale for m in range(K)], dim=0)
        super().__init__(locs, covs)


class RingModel(PyroModule):
    def __init__(self, name, radius=10, scale=0.5, K=8):
        super().__init__()
        self.name, self.radius, self.scale, self.K = name, radius, scale, K
        self.dist = RingGMM(self.radius, self.scale, self.K)

    def forward(self):
        return pyro.sample(self.name, self.dist)


def ring_gmm(name, radius=10, scale=0.5, K=8):
    return primitive(RingModel(name, radius, scale, K))


class ResMLPJ(PyroModule):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.map_joint = nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.ReLU())
        self.map_mu = nn.Sequential(nn.Linear(dim_hidden, dim_out))
        self.map_cov = nn.Sequential(nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        y = self.map_joint(x)
        mu = self.map_mu(y) + x
        cov_emb = self.map_cov(y)
        return torch.cat((mu, cov_emb), dim=-1)

    def initialize_(self, loc_offset, cov_emb):
        nn.init.zeros_(self.map_mu[0].weight)
        nn.init.zeros_(self.map_mu[0].bias)
        self.map_mu[0].bias.data.add_(loc_offset)

        nn.init.zeros_(self.map_cov[0].weight)
        nn.init.zeros_(self.map_cov[0].bias)
        self.map_cov[0].bias.data.add_(cov_emb)


class MultivariateNormal(PyroModule):
    def __init__(self, name, loc=torch.zeros(1), cov=torch.ones(1)):
        super().__init__()
        self.name, self.loc, self.cov = name, loc, cov
        self.dist = dist.MultivariateNormal(loc=self.loc, covariance_matrix=self.cov)

    def forward(self):
        return pyro.sample(self.name, self.dist)


def multivariate_normal(name, loc, cov):
    return primitive(MultivariateNormal(name, loc, cov))


class FixedMultivariateNormal(MultivariateNormal):
    def __init__(self, name, loc=torch.zeros(1), cov=torch.ones(1)):
        super().__init__(name, loc, cov)
        self.name, self.loc, self.cov = name, loc, cov
        self.loc.requires_grad_(False)
        self.cov.requires_grad_(False)


def fixed_multivariate_normal(name, loc, cov):
    return primitive(FixedMultivariateNormal(name, loc, cov))


class ConditionalMultivariateNormalKernel(PyroModule):
    def __init__(self, name: str, loc: Tensor, cov: Tensor, net):
        super().__init__()
        self.name = name
        self.dim_in = 2
        self.cov_dim = cov.shape[0]
        self.cov_emb = cov.diag().expm1().log()
        self.net = net
        self.net.initialize_(torch.zeros_like(loc), self.cov_emb)

    def forward(self, ext_from):
        out = self.net(ext_from)
        mu, cov_emb = torch.split(out, [2, 2], dim=-1)
        cov = torch.diag_embed(F.softplus(cov_emb))
        return pyro.sample(
            self.name, dist.MultivariateNormal(loc=mu, covariance_matrix=cov)
        )


def conditional_multivariate_normal_kernel(name, loc, cov, net):
    return primitive(ConditionalMultivariateNormalKernel(name, loc, cov, net))


class Tempered(PyroModule):
    def __init__(
        self,
        density1,
        density2,
        beta: Tensor,
        event_shape=torch.Size([]),
        optimize=False,
    ):
        super().__init__()
        assert isinstance(density1, type(density2)), "densities must be the same type"
        assert torch.all(beta > 0.0) and torch.all(
            beta < 1.0
        ), "tempered densities are β=(0, 1) for clarity. Use model directly for β=0 or β=1"
        self._event_shape = (
            density1.event_shape if "event_shape" in dir(density1) else event_shape
        )
        self.optimize = optimize
        self.logit = (
            PyroParam(torch.logit(beta)) if self.optimize else torch.logit(beta)
        )
        self.density1, self.density2 = density1, density2
        self.beta = beta

    def log_prob(self, value):
        def _log_prob(g, value):
            if isinstance(g, primitive):
                if "log_prob" in dir(g.program):
                    return g.program.log_prob(value)
                elif "dist" in dir(g.program):
                    return g.program.dist.log_prob(value)
            breakpoint()
            raise RuntimeError("not implemented")

        return _log_prob(self.density1, value) * (
            1 - torch.sigmoid(self.logit)
        ) + _log_prob(self.density2, value) * torch.sigmoid(self.logit)


class TemperedModel(PyroModule):
    def __init__(self, name, d1, d2, beta: Tensor, optimize=False):
        super().__init__()
        self.name = name
        self.d1, self.d2, self.beta, self.optimize = d1, d2, beta, optimize
        self.dist = Tempered(self.d1, self.d2, self.beta, self.optimize)

    def forward(self):
        return pyro.sample(self.name, self.dist)


def tempered(name, d1, d2, beta, optimize=False):
    return primitive(TemperedModel(name, d1, d2, beta, optimize))


def anneal_between(left, right, total_num_targets, optimize_path=False):
    # Make an annealing path
    betas = torch.arange(0.0, 1.0, 1.0 / (total_num_targets - 1))[1:]  # g_0 is beta=0
    path = [
        tempered(f"g{k}", left, right, beta, optimize=optimize_path)
        for k, beta in zip(range(1, total_num_targets - 1), betas)
    ]
    path = [left] + path + [right]
    assert len(path) == total_num_targets  # sanity check that the betas line up
    return path


def paper_model(num_targets=8, optimize_path=False):
    """model defined in 'Learning Proposals for Probabilistic Programs with Inference Combinators'"""
    g0 = fixed_multivariate_normal(
        name="g0",
        loc=torch.zeros(2),
        cov=torch.eye(2) * 5 ** 2,
    )
    gK = ring_gmm(radius=10, scale=0.5, K=8, name=f"g{num_targets - 1}")

    def paper_kernel(to_: int, std: float, reparameterized: bool):
        return conditional_multivariate_normal_kernel(
            name=f"g{to_}",
            loc=torch.zeros(2),
            cov=torch.eye(2) * std ** 2,
            net=ResMLPJ(dim_in=2, dim_hidden=50, dim_out=2),
        )

    return dict(
        targets=anneal_between(g0, gK, num_targets, optimize_path=optimize_path),
        forwards=[
            paper_kernel(to_=i + 1, std=1.0, reparameterized=True)
            for i in range(num_targets - 1)
        ],
        reverses=[
            paper_kernel(to_=i, std=1.0, reparameterized=False)
            for i in range(num_targets - 1)
        ],
    )


@nested_objective
def nvo_avo(p_out, q_out, lw, lv, sample_dims=-1) -> Tensor:
    nw = F.softmax(torch.zeros_like(lv), dim=sample_dims)
    loss = (nw * (-lv)).sum(dim=(sample_dims,), keepdim=False)
    return loss


@nested_objective
def nvo_rkl_mod(p_out, q_out, lw, lv, sample_dim=-1) -> Tensor:
    """
    Metric: KL(g_k-1(z_k-1)/Z_k-1 q_k(z_k | z_k-1) || g_k(z_k)/Z_k-1 r_k(z_k-1 | z_k)) = Exp[-(dlZ + lv)]
    """
    _eval0 = lambda e: e - e.detach()
    _eval1 = lambda e: torch.exp(_eval0(e))

    def _eval_nrep(node):
        generator = node["fn"]
        if isinstance(generator, Distribution):
            out = node.copy()
            out["value"] = node["value"].detach()
            # out.reparameterized = False # FIXME not sure how to enforce this in pyro
            return out
        elif getattr(
            generator, "log_prob"
        ):  # not a normalized distribution, but still has a log_prob
            out = node.copy()
            out["value"] = node["value"].detach()
            return out
        else:
            raise NotImplementedError(
                "Only supports RandomVariable and ImproperRandomVariable"
            )

    def _estimate_mc(
        values: Tensor,
        log_weights: Tensor,
        sample_dims: Tuple[int],
        reducedims: Tuple[int],
        keepdims: bool,
    ) -> Tensor:
        nw = F.softmax(log_weights, dim=sample_dims)
        return (nw * values).sum(dim=reducedims, keepdim=keepdims)

    reducedims = (sample_dim,)

    lw = lw  # already detached
    lv = lv
    proposal_trace = q_out
    target_trace = p_out

    proposal_trace.compute_log_prob()
    target_trace.compute_log_prob()

    q_sites = {
        k for k, n in p_out.nodes.items() if is_sample_type(n) and is_auxiliary(n)
    }
    q_extended_sites = {
        k for k, n in p_out.nodes.items() if is_sample_type(n) and not is_auxiliary(n)
    }

    # TODO: in probtorch we fix an index, the following code still requires this assumption
    assert len(q_sites) == 1 and len(q_sites) == len(q_extended_sites)
    q_site = list(q_sites)[0]
    q_ext_site = list(q_extended_sites)[0]

    # rv_proposal = proposal_trace["g{}".format(out.ix)]
    rv_proposal = proposal_trace.nodes[q_site]
    # rv_target = target_trace["g{}".format(out.ix + 1)]
    rv_target = target_trace.nodes[q_ext_site]
    # rv_fwd = proposal_trace["g{}".format(out.ix + 1)]
    rv_fwd = proposal_trace.nodes[q_ext_site]
    # rv_rev = target_trace["g{}".format(out.ix)]
    rv_rev = target_trace.nodes[q_site]

    # ldZ = Z_{k} / Z_{k-1}
    # Chat with Hao --> estimating dZ might be instable when resampling
    ldZ = lv.detach().logsumexp(dim=sample_dim) - math.log(lv.shape[sample_dim])
    # TODO: set ldZ to 0 if RandomVariable similar to grad  terms
    f = -(lv - ldZ)

    grad_log_Z1_term = (
        _estimate_mc(
            rv_proposal["log_prob"],
            lw,
            sample_dims=sample_dim,
            reducedims=reducedims,
            keepdims=False,
        )
        if not isinstance(rv_proposal["fn"], Distribution)
        else torch.tensor(0.0)
    )
    grad_log_Z2_term = (
        _estimate_mc(
            _eval_nrep(rv_target)["log_prob"],
            lw + lv.detach(),
            sample_dims=sample_dim,
            reducedims=reducedims,
            keepdims=False,
        )
        if not isinstance(rv_target["fn"], Distribution)
        else torch.tensor(0.0)
    )

    # FIXME: rewrite this
    if getattr(rv_proposal["fn"], "has_rsample", False):
        # Compute reparameterized gradient
        kl_term = _estimate_mc(
            f,
            lw,
            sample_dims=sample_dim,
            reducedims=reducedims,
            keepdims=False,
        )
        return kl_term - _eval0(grad_log_Z1_term) + _eval0(grad_log_Z2_term)

    baseline = _estimate_mc(
        f.detach(),
        lw,
        sample_dims=sample_dim,
        reducedims=reducedims,
        keepdims=False,
    )

    kl_term = _estimate_mc(
        (f - baseline) * _eval1(rv_proposal["log_prob"] - grad_log_Z1_term)
        - _eval0(rv_proposal["log_prob"]),
        lw,
        sample_dims=sample_dim,
        reducedims=reducedims,
        keepdims=False,
    )

    loss = kl_term + _eval0(grad_log_Z2_term) + baseline

    return loss


def vsmc(
    targets,
    forwards,
    reverses,
    loss_fn,
    resample=False,
    stl=False,
    batch_dim=None,
    debug=False,
):
    q = targets[0]
    for ix, (fwd, rev, p) in enumerate(zip(forwards, reverses, targets[1:])):
        q = propose(
            p=extend(p, rev),
            q=compose(fwd, q),
            loss_fn=loss_fn,
            q_prev=q if debug else None,
        )
        if stl:
            q = augment_logweight(q, pre=stl_trick)
        if resample and ix < len(forwards) - 1:
            # NOTE: requires static knowledge that sample_dim is first!
            assert batch_dim is None or batch_dim in {-1, 1}
            q = combinator.resample(q, batch_dim=batch_dim, sample_dim=0)
    return q


def effective_sample_size(lw: Tensor, sample_dims=-1) -> Tensor:
    lnw = F.softmax(lw, dim=sample_dims).log()
    ess = torch.exp(-torch.logsumexp(2 * lnw, dim=sample_dims))
    return ess


def log_Z_hat(lw: Tensor, sample_dims=-1) -> Tensor:
    return Z_hat(lw, sample_dims=sample_dims).log()


def Z_hat(lw: Tensor, sample_dims=-1) -> Tensor:
    return lw.exp().mean(dim=sample_dims)


def save_models(models, filename, weights_dir="./weights") -> None:
    checkpoint = {k: v.program.state_dict() for k, v in models.items()}
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    torch.save(checkpoint, f"{weights_dir}/{filename}")


def load_models(model, filename, weights_dir="./weights", **kwargs) -> None:
    checkpoint = torch.load(f"{weights_dir}/{filename}", **kwargs)
    {k: v.load_state_dict(checkpoint[k]) for k, v in model.items()}


def models_as_dict(model_iter, names):
    assert isinstance(model_iter, list) or all(
        map(lambda ms: isinstance(ms, list), model_iter.values())
    ), "takes a list or dict of lists"
    assert len(names) == len(model_iter), "names must exactly align with model lists"
    model_dict = dict()
    for i, (name, models) in enumerate(
        zip(names, model_iter) if isinstance(model_iter, list) else model_iter.items()
    ):
        for j, m in enumerate(models):
            model_dict[f"{str(i)}_{name}_{str(j)}"] = m
    return model_dict


def save_annealing_model(targets, forwards, reverses, filename=None):
    assert filename is not None
    save_models(
        models_as_dict(
            [targets, forwards, reverses], ["targets", "forwards", "reverses"]
        ),
        filename="{}.pt".format(filename),
        weights_dir="./weights",
    )


def load_annealing_model(targets, forwards, reverses, filename=None):
    assert filename is not None
    load_models(
        models_as_dict(
            [targets, forwards, reverses], ["targets", "forwards", "reverses"]
        ),
        filename="./{}.pt".format(filename),
        weights_dir="./weights",
    )


def mkfilename(K, iterations, seed, loss_fn, S=288, resample=True, optimize_path=False):
    return "nvi{}{}_{}_S{}_K{}_I{}_seed{}".format(
        "r" if resample else "",
        "s" if optimize_path else "",
        loss_fn,
        S,
        K,
        iterations,
        seed,
    )


def train_and_test_avo(
    mkSummaryWriter,
    num_targets,
    smoke_test_level=0,
    seed=8,
    debug=False,
    resample=False,
    stl=False,
    optimize_path=False,
    objective="nvo_avo",
):
    assert smoke_test_level < 5 and smoke_test_level >= 0
    assert objective in {"nvo_rkl", "nvo_avo"}
    S = 288
    K = num_targets
    loss_fn_name = objective
    loss_fn = nvo_avo if loss_fn_name == "nvo_avo" else nvo_rkl_mod
    iterations = (
        smoke_test_level * smoke_test_level * 100 if smoke_test_level > 0 else 20_000
    )
    filename = mkfilename(K, iterations, seed, loss_fn_name, S, resample, optimize_path)

    pyro.set_rng_seed(seed)
    model = paper_model(num_targets=num_targets, optimize_path=optimize_path)
    targets, forwards, reverses = [
        model[k] for k in ["targets", "forwards", "reverses"]
    ]
    infer = vsmc(
        targets=targets,
        forwards=forwards,
        reverses=reverses,
        loss_fn=loss_fn,
        resample=resample,
        stl=stl,
        debug=debug,
    )
    optimizer = torch.optim.Adam(
        params=[dict(params=x.program.parameters()) for x in forwards + reverses],
    )
    Z = targets[-1].program.K
    lZ = math.log(Z)
    bar = trange(iterations)
    L = int(S / num_targets)
    writer = mkSummaryWriter(
        comment=":"
        + "+".join(
            [
                f"K{K}",
                f"s{seed}",
                loss_fn_name,
                "stl" if stl else "",
                "resample" if resample else "",
                "opt" if optimize_path else "",
            ]
        )
    )
    for ix in bar:
        optimizer.zero_grad()
        with pyro.plate("samples", L):
            out = infer()
            loss = out.nodes[_LOSS]["value"]
            loss.backward()
            lw = out.nodes["_LOGWEIGHT"]["value"]
            ess = effective_sample_size(lw).item()
            lZh = log_Z_hat(lw).item()
            if debug:
                seq_trace = infer.debug
                breakpoint()
            bar.set_postfix_str(
                "ESS={: <6} / {}, lZhat={: <5}/{:.2f}".format(
                    "{: .3f}".format(ess / L), 1, "{: .2f}".format(lZh), lZ
                )
            )
            writer.add_scalar("ESS/train", ess / L, ix)
            writer.add_scalar("log_z_hat/train", lZh, ix)
        optimizer.step()
    writer.flush()

    # Metrics of logˆZ and effective sample size average over 100 batches of 1,000 samples
    test_batches = smoke_test_level * 10 if smoke_test_level > 0 else 100
    test_samples = smoke_test_level * 100 if smoke_test_level > 0 else 1000
    with pyro.plate("batches", test_batches):
        with pyro.plate("samples", test_samples):
            metric_samples = vsmc(
                targets=targets,
                forwards=forwards,
                reverses=reverses,
                loss_fn=nvo_avo,
                resample=False,
                stl=False,
                batch_dim=-1,
            )()
    lws = metric_samples.nodes["_LOGWEIGHT"]["value"]
    esss = effective_sample_size(lws, sample_dims=0)
    lZhs = log_Z_hat(lws, sample_dims=0)
    print("[K={}] lws shape: {}".format(K, "x".join(map(str, lws.shape))))
    _ess = esss.mean().item()
    writer.add_scalar("ESS/test", _ess, 0)
    print("[K={}] {}-avg ess: {:.2f} / {}".format(K, test_batches, _ess, test_samples))
    _lZh = lZhs.mean().item()
    writer.add_scalar("log_z_hat/test", _lZh, 0)
    print("[K={}] {}-avg lZh: {:.2f} / {:.2f}".format(K, test_batches, _lZh, lZ))
    return _ess, _lZh, model, filename


class NoopWriter(SummaryWriter):
    def __init__(self):
        pass

    def add_scalar(self, a, b, c):
        pass

    def flush(self):
        pass


def test_avo_2step():
    esss, lZhs = [], []
    for seed in range(10):
        _ess, _lZh, _, _ = train_and_test_avo(
            NoopWriter, num_targets=2, smoke_test_level=0, seed=seed
        )
        esss.append(_ess)
        lZhs.append(_lZh)
    esss, lZhs = torch.tensor(esss), torch.tensor(lZhs)
    assert esss.mean().item() > 300, "ess is less than expected"
    breakpoint()
    assert esss.std().item() < 20, "ess is a bit too variedd"
    assert abs(lZhs.mean().item() - 1.88) < 0.1, "log_Z_hat is off"


@mark.skip("TODO comment this and adjust test as needed")
def test_avo_4step_grads_dont_leak():
    seed = 4
    pyro.set_rng_seed(seed)
    model = paper_model(num_targets=4, optimize_path=False)
    targets, forwards, reverses = [
        model[k] for k in ["targets", "forwards", "reverses"]
    ]
    L = int(288 / 4)
    q0 = targets[0]
    [(f01, r10, q1), (f12, r21, q2), (f23, r32, p)] = list(
        zip(forwards, reverses, targets[1:])
    )

    @nested_objective
    def nvo_avo1(p_out, q_out, lw, lv, sample_dims=-1) -> Tensor:
        return (-lv).sum(dim=(sample_dims,), keepdim=False)

    @nested_objective
    def nvo_avo2(p_out, q_out, lw, lv, sample_dims=-1) -> Tensor:
        return (-lv).sum(dim=(sample_dims,), keepdim=False)

    @nested_objective
    def nvo_avo3(p_out, q_out, lw, lv, sample_dims=-1) -> Tensor:
        loss = (-lv).sum(dim=(sample_dims,), keepdim=False)
        loss.backward()
        return loss

    q = propose(p=extend(q1, r10), q=compose(f01, q0), loss_fn=nvo_avo1, q_prev=None)
    q = propose(p=extend(q2, r21), q=compose(f12, q), loss_fn=nvo_avo2, q_prev=q)
    q = propose(p=extend(p, r32), q=compose(f23, q), loss_fn=nvo_avo3, q_prev=q)

    infer = q
    optimizer = torch.optim.Adam(
        params=[dict(params=x.program.parameters()) for x in forwards + reverses],
    )
    Z = targets[-1].program.K
    lZ = math.log(Z)

    optimizer.zero_grad()

    params_pre1 = [param for k in [f01, r10] for param in k.program.parameters()]
    params_pre2 = [param for k in [f12, r21] for param in k.program.parameters()]
    params_pre3 = [param for k in [f23, r32] for param in k.program.parameters()]

    with pyro.plate("samples", L):
        out = infer()
        loss = out.nodes[_LOSS]["value"]
        loss.backward()
        lw = out.nodes["_LOGWEIGHT"]["value"]
        ess = effective_sample_size(lw).item()
        lZh = log_Z_hat(lw).item()
        optimizer.step()

    params_post1 = [param for k in [f01, r10] for param in k.program.parameters()]
    params_post2 = [param for k in [f12, r21] for param in k.program.parameters()]
    params_post3 = [param for k in [f23, r32] for param in k.program.parameters()]

    breakpoint()
    print()


def main(
    smoke_test_level=0,
    num_targets=8,
    with_tb=True,
    seed=3,
    debug=False,
    resample=False,
    stl=False,
    optimize_path=False,
    objective=None,
):
    mkSummaryWriter = SummaryWriter if with_tb else NoopWriter
    _ess, _lZh, model, filename = train_and_test_avo(
        mkSummaryWriter,
        num_targets=num_targets,
        smoke_test_level=smoke_test_level,
        seed=seed,
        debug=debug,
        resample=resample,
        stl=stl,
        optimize_path=optimize_path,
        objective=objective,
    )
    targets, forwards, reverses = [
        model[k] for k in ["targets", "forwards", "reverses"]
    ]

    if _ess > 0:
        num_renderable_samples = 100000
        chain = reduce(lambda q, fwd: compose(fwd, q), forwards, targets[0])
        target = targets[-1]
        with pyro.plate("render", num_renderable_samples):
            chain_samples = chain()
            plot_sequence(
                [
                    (f"g{k}", chain_samples.nodes[f"g{k}"]["value"].detach())
                    for k in range(num_targets)
                ]
                + [("target", target().nodes["_RETURN"]["value"])],
                path=f"{filename}_sequence.png",
            )

        if smoke_test_level == 0:
            save_annealing_model(targets, forwards, reverses, filename=filename)
            torch.save(
                (_ess, _lZh),
                "./metrics/{}-metric-tuple_S{}_B{}-ess-logZhat.pt".format(
                    filename, 1000, 100
                ),
            )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clear",
        help="clear output artifacts before run",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--debug",
        help="clear output artifacts before run",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--optimize_path",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--resample",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--stl",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--objective", default="nvo_avo", choices=["nvo_avo", "nvo_rkl"]
    )
    parser.add_argument("--seed", help="set seed", type=int, default=3)
    parser.add_argument("--smoketest", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--targets", type=int, choices=[2, 4, 6, 8])
    args = parser.parse_args()
    if args.clear:
        try:
            shutil.rmtree("./runs")
        except FileNotFoundError:
            pass

    print(f"K={args.targets}")
    start = time.time()
    main(
        smoke_test_level=args.smoketest,
        num_targets=args.targets,
        seed=args.seed,
        debug=args.debug,
        objective=args.objective,
    )
    end = time.time()
    print("Time consumed in working: {:.1f}s".format(end - start))

# AVO:
#         N |   2  |   4  |   6  |   8
#           ,------+------+------+------
# log Z hat | 1.88 | 1.99 | 2.05 | 2.06
#       ESS |  426 |  291 |  285 |  295
