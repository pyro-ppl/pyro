import torch

import pyro
import pyro.distributions as dist


def _densify_shocks(length, times, sparse_shocks):
    with torch.no_grad():
        lb = times.floor().clamp_(min=0, max=length)
        ub = times.ceil().clamp_(min=0, max=length)
        is_degenerate = (lb == ub)
    lb_weight = ub - time
    lb_weight.data[is_degenerate] = 0.5
    ub_weight = time - lb
    ub_weight.data[is_degenerate] = 0.5

    dense_shocks = sparse_shocks.new_zeros((length + 1,) + shocks.shape[1:])
    dense_shocks.scatter_add_(0, lb, lb_weights * sparse_shocks)
    dense_shocks.scatter_add_(0, ub, ub_weights * sparse_shocks)
    return dense_shocks[:-1]


class StableHMMReparam:
    """
    Levy-Ito decomposition of a :class:`~pyro.distributions.hmm.StableHMM` .
    """
    def __init__(self, num_jumps):
        self.num_jumps = num_jumps

    def __call__(self, name, fn, obs):
        assert isinstance(fn, dist.StableHMM)

        # Sample shocks to initial state.
        init_shocks = pyro.sample("{}_init_shocks".format(name),
                                  dist.Pareto(1, self.stability))
        init_dist = dist.Normal(self.init_dist.scale,
                                self.init_dist.loc + init_shocks).to_event(1)

        # Sample shocks to latent state.
        length = self.length
        shock_times = pyro.sample("{}_trans_shock_times".format(name),
                                  dist.Uniform(0, length).to_event(2))
        # FIXME we need both positive and negative shocks.
        shock_sizes = pyro.sample("{}_trans_shock_sizes".format(name),
                                  dist.Pareto(1, self.stability).to_event(2))
        trans_shocks = _densify_shocks(length, shock_times, shock_sizes)
        trans_dist = dist.Normal(self.trans_dist.scale,
                                 self.trans_dist.loc + trans_shocks).to_event(1)

        # Sample shocks to observations.
        # FIXME we need both positive and negative shocks.
        obs_shocks = pyro.sample("{}_obs_shocks".format(name),
                                 dist.Pareto(1, self.stability).to_event(2))
        obs_dist = dist.Normal(self.obs_dist.scale,
                               self.obs_dist.loc + obs_shocks).to_event(1)

        hmm = dist.GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist)
        return hmm, obs
