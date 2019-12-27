import torch

import pyro
import pyro.distributions as dist

from .reparam import Reparam
from .stable import SymmetricStableReparam


def _densify_shocks(length, times, sparse_shocks):
    with torch.no_grad():
        lb = times.floor().clamp_(min=0, max=length)
        ub = times.ceil().clamp_(min=0, max=length)
        is_degenerate = (lb == ub)
    lb_weight = ub - times
    lb_weight.data[is_degenerate] = 0.5
    ub_weight = times - lb
    ub_weight.data[is_degenerate] = 0.5

    dense_shocks = sparse_shocks.new_zeros((length + 1,) + sparse_shocks.shape[1:])
    dense_shocks.scatter_add_(0, lb, lb_weight * sparse_shocks)
    dense_shocks.scatter_add_(0, ub, ub_weight * sparse_shocks)
    return dense_shocks[:-1]


class StableHMMReparam(Reparam):
    """
    Levy-Ito decomposition of :class:`~pyro.distributions.StableHMM` random
    variables whose ``initial_dist`` and ``observation_dist`` are symmetric
    (but whose ``transition_dist`` may be skewed).

    This is useful for training the parameters of a
    :class:`~pyro.distributions.StableHMM` distribution, whose
    :meth:`~pyro.distributions.StableHMM.log_prob` method is undefined.

    This introduces auxiliary random variables conditioned on which the
    remaining process is a :class:`~pyro.distributions.GaussianHMM` .  The
    initial and observation distributions are reparameterized by
    :class:`~pyro.infer.reparam.stable.SymmetricStableReparam` . The latent
    transition process is Levy-Ito decomposed into drift + Brownian motion + a
    compound Poisson process; we neglect the generalized compensated Poisson
    process of small shocks, and approximate the compound Poisson process with
    a fixed ``num_shocks`` over the time interval of interest. As
    ``num_shocks`` increases, the cutoff on shock size tends to zero and
    approximation becomes exact.

    :param int num_shocks: Fixed number of shocks in the compound Poisson
        process component of the Levy-Ito decomposition of latent state.
    """
    def __init__(self, num_shocks):
        assert isinstance(num_shocks, int) and num_shocks > 0
        self.num_shocks = num_shocks

    def __call__(self, name, fn, obs):
        assert isinstance(fn, dist.StableHMM)
        hidden_dim = fn.hidden_dim
        duration = fn.transition_dist.batch_shape[-1]
        stability = fn.initial_dist.base_dist.stability[..., 0]

        # Sample positive and negative shocks to latent state,
        # independent over each hidden dim.
        shock_times = pyro.sample("{}_shock_times".format(name),
                                  dist.Uniform(0, duration)
                                      .expand([2, self.num_shocks, hidden_dim])
                                      .to_event(3))
        shock_sizes = pyro.sample("{}_shock_sizes".format(name),
                                  dist.Pareto(1, stability.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                                      .expand([2, self.num_shocks, hidden_dim])
                                      .to_event(3))
        pos_shock_times, neg_shock_times = shock_times.unbind(-3)
        pos_shock_sizes, neg_shock_sizes = shock_sizes.unbind(-3)
        pos_shocks = _densify_shocks(duration, pos_shock_times, pos_shock_sizes)
        neg_shocks = _densify_shocks(duration, neg_shock_times, neg_shock_sizes)
        # FIXME correctly scale pos_shocks and neg_shocks
        shocks = pos_shocks - neg_shocks
        # FIXME correct scale and loc
        trans_dist = dist.Normal(self.trans_dist.scale,
                                 self.trans_dist.loc + shocks).to_event(1)

        # Reparameterize the initial distribution as conditionally Gaussian.
        init_dist, _ = SymmetricStableReparam()("{}_init".format(name), fn.initial_dist, None)
        assert isinstance(init_dist, dist.Independent)
        assert isinstance(init_dist.base_dist, dist.Normal)

        # Reparameterize the observation distribution as conditionally Gaussian.
        obs_dist, obs = SymmetricStableReparam()("{}_obs".format(name), fn.observation_dist, obs)
        assert isinstance(obs_dist, dist.Independent)
        assert isinstance(obs_dist.base_dist, dist.Normal)

        # Reparameterize the entire HMM as conditionally Gaussian.
        hmm = dist.GaussianHMM(init_dist, fn.trans_mat, trans_dist.to_event(1),
                               fn.obs_mat, obs_dist)
        return hmm, obs
