import math

import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.stable import _standard_stable, _unsafe_standard_stable
from pyro.infer.util import is_validation_enabled

from .reparam import Reparam


class StableReparam(Reparam):
    """
    Auxiliary variable reparameterizer for
    :class:`~pyro.distributions.Stable` latent variables.

    This is useful in inference of latent :class:`~pyro.distributions.Stable`
    variables because the :meth:`~pyro.distributions.Stable.log_prob` is not
    implemented.

    This uses the Chambers-Mallows-Stuck method [1], creating a pair of
    parameter-free auxiliary distributions (``Uniform(-pi/2,pi/2)`` and
    ``Exponential(1)``) with well-defined ``.log_prob()`` methods, thereby
    permitting use of reparameterized stable distributions in likelihood-based
    inference algorithms like SVI and MCMC.

    This reparameterization works only for latent variables, not likelihoods.
    For likelihood-compatible reparameterization see
    :class:`SymmetricStableReparam` .

    [1] J.P. Nolan (2017).
        Stable Distributions: Models for Heavy Tailed Data.
        http://fs2.american.edu/jpnolan/www/stable/chap1.pdf
    """
    def __call__(self, name, fn, obs):
        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.Stable)
        assert obs is None, "StableReparam does not support observe statements"

        # Draw parameter-free noise.
        proto = fn.stability
        half_pi = proto.new_full(proto.shape, math.pi / 2)
        one = proto.new_ones(proto.shape)
        u = pyro.sample("{}_uniform".format(name),
                        self._wrap(dist.Uniform(-half_pi, half_pi), event_dim))
        e = pyro.sample("{}_exponential".format(name),
                        self._wrap(dist.Exponential(one), event_dim))

        # Differentiably transform.
        x = _standard_stable(fn.stability, fn.skew, u, e)
        value = fn.loc + fn.scale * x

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(value, event_dim=event_dim).mask(False)
        return new_fn, value


class SymmetricStableReparam(Reparam):
    """
    Auxiliary variable reparameterizer for symmetric
    :class:`~pyro.distributions.Stable` random variables (i.e. those for which
    ``skew=0``).

    This is useful in inference of symmetric
    :class:`~pyro.distributions.Stable` variables because the
    :meth:`~pyro.distributions.Stable.log_prob` is not implemented.

    This reparameterizes a symmetric :class:`~pyro.distributions.Stable` random
    variable as a totally-skewed (``skew=1``)
    :class:`~pyro.distributions.Stable` scale mixture of
    :class:`~pyro.distributions.Normal` random variables. See Proposition 3. of
    [1] (but note we differ since :class:`Stable` uses Nolan's continuous S0
    parameterization).

    [1] Alvaro Cartea and Sam Howison (2009)
        "Option Pricing with Levy-Stable Processes"
        https://pdfs.semanticscholar.org/4d66/c91b136b2a38117dd16c2693679f5341c616.pdf
    """
    def __call__(self, name, fn, obs):
        fn, event_dim = self._unwrap(fn)
        assert isinstance(fn, dist.Stable)
        if is_validation_enabled():
            if not (fn.skew == 0).all():
                raise ValueError("SymmetricStableReparam found nonzero skew")
            if not (fn.stability < 2).all():
                raise ValueError("SymmetricStableReparam found stability >= 2")

        # Draw parameter-free noise.
        proto = fn.stability
        half_pi = proto.new_full(proto.shape, math.pi / 2)
        one = proto.new_ones(proto.shape)
        u = pyro.sample("{}_uniform".format(name),
                        self._wrap(dist.Uniform(-half_pi, half_pi), event_dim))
        e = pyro.sample("{}_exponential".format(name),
                        self._wrap(dist.Exponential(one), event_dim))

        # Differentiably transform to scale drawn from a totally-skewed stable variable.
        _, z = _unsafe_standard_stable(fn.stability / 2, 1, u, e)
        assert (z >= 0).all()
        scale = fn.scale * (2 ** 0.5) * (math.pi / 4 * fn.stability).cos().pow(1 / fn.stability) * z.sqrt()
        scale = scale.clamp(min=torch.finfo(scale.dtype).tiny)

        # Construct a scaled Gaussian.
        new_fn = self._wrap(dist.Normal(fn.loc, scale), event_dim)
        return new_fn, obs


def _densify_jumps(duration, times, sparse_jumps):
    with torch.no_grad():
        lb = times.floor().long().clamp_(min=0, max=duration)
        ub = times.ceil().long().clamp_(min=0, max=duration)
        is_degenerate = (lb == ub)
    lb_weight = ub - times
    lb_weight.data[is_degenerate] = 0.5
    ub_weight = times - lb
    ub_weight.data[is_degenerate] = 0.5

    shape = list(sparse_jumps.shape)
    shape[-2] = duration + 1
    dense_jumps = sparse_jumps.new_zeros(torch.Size(shape))
    dense_jumps.scatter_add_(-2, lb, lb_weight * sparse_jumps)
    dense_jumps.scatter_add_(-2, ub, ub_weight * sparse_jumps)
    return dense_jumps[..., :-1, :]


class StableHMMReparam(Reparam):
    """
    Approximate Levy-Ito decomposition of
    :class:`~pyro.distributions.StableHMM` random variables whose
    ``initial_dist`` and ``observation_dist`` are symmetric (but whose
    ``transition_dist`` may be skewed).

    This is useful for training the parameters of a
    :class:`~pyro.distributions.StableHMM` distribution, whose
    :meth:`~pyro.distributions.StableHMM.log_prob` method is undefined.

    This introduces auxiliary random variables conditioned on which the process
    becomes a :class:`~pyro.distributions.GaussianHMM` . The initial and
    observation distributions are reparameterized by
    :class:`SymmetricStableReparam` . The latent transition process is Levy-Ito
    decomposed into drift + Brownian motion + a compound Poisson process (see
    [1] section 1.2.6). We neglect the generalized compensated Poisson process
    of small jumps, and approximate the compound Poisson process with a fixed
    ``num_jumps`` over the time interval of interest. As ``num_jumps``
    increases, the jump size cutoffs tend to zero and approximation improves.

    [1] Andreas E. Kyprianou (2013)
        "Fluctuations of Levy Processes with Applications"
        https://people.bath.ac.uk/ak257/book2e/book2e.pdf

    :param int num_jumps: Fixed number of jumps to approximate the compound
        Poisson process component of the Levy-Ito decomposition of latent
        state. This automatically determines the minimum jump size cutoffs.
    """
    def __init__(self, num_jumps):
        assert isinstance(num_jumps, int) and num_jumps > 0
        self.num_jumps = num_jumps

    def __call__(self, name, fn, obs):
        assert isinstance(fn, dist.StableHMM)
        hidden_dim = fn.hidden_dim
        duration = fn.transition_dist.batch_shape[-1]
        stability = fn.initial_dist.base_dist.stability[..., 0]

        # Sample positive and negative jumps to latent state,
        # independent over each hidden dim.
        jump_times = pyro.sample("{}_jump_times".format(name),
                                 dist.Uniform(0, duration)
                                     .expand([2, self.num_jumps, hidden_dim])
                                     .to_event(3))
        jump_sizes = pyro.sample("{}_jump_sizes".format(name),
                                 dist.Pareto(1, stability.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                                     .expand([2, self.num_jumps, hidden_dim])
                                     .to_event(3))
        pos_jump_times, neg_jump_times = jump_times.unbind(-3)
        pos_jump_sizes, neg_jump_sizes = jump_sizes.unbind(-3)
        pos_jumps = _densify_jumps(duration, pos_jump_times, pos_jump_sizes)
        neg_jumps = _densify_jumps(duration, neg_jump_times, neg_jump_sizes)
        # FIXME correctly scale pos_jumps and neg_jumps
        jumps = pos_jumps - neg_jumps
        assert jumps.shape[-2:] == (duration, hidden_dim)
        # FIXME correct scale and loc
        trans_dist = fn.transition_dist.base_dist
        trans_dist = dist.Normal(trans_dist.loc + jumps,
                                 trans_dist.scale).to_event(1)

        # Reparameterize the initial distribution as conditionally Gaussian.
        init_dist, _ = SymmetricStableReparam()("{}_init".format(name), fn.initial_dist, None)
        assert isinstance(init_dist, dist.Independent)
        assert isinstance(init_dist.base_dist, dist.Normal)

        # Reparameterize the observation distribution as conditionally Gaussian.
        obs_dist = fn.observation_dist.base_dist
        obs_dist, obs = SymmetricStableReparam()("{}_obs".format(name), obs_dist.to_event(2), obs)
        assert isinstance(obs_dist, dist.Independent)
        assert isinstance(obs_dist.base_dist, dist.Normal)
        obs_dist = obs_dist.base_dist.to_event(1)

        # Reparameterize the entire HMM as conditionally Gaussian.
        hmm = dist.GaussianHMM(init_dist, fn.transition_matrix, trans_dist,
                               fn.observation_matrix, obs_dist)
        return hmm, obs
