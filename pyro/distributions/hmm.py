# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property

from pyro.distributions.torch import Categorical, Gamma, Independent, MultivariateNormal
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape
from pyro.ops.gamma_gaussian import (GammaGaussian, gamma_and_mvn_to_gamma_gaussian, gamma_gaussian_tensordot,
                                     matrix_and_mvn_to_gamma_gaussian)
from pyro.ops.gaussian import Gaussian, gaussian_tensordot, matrix_and_mvn_to_gaussian, mvn_to_gaussian


@torch.jit.script
def _linear_integrate(init, trans, shift):
    """
    Integrate the inhomogeneous linear shifterence equation::

        x[0] = init
        x[t] = x[t-1] @ trans[t] + shift[t]

    :return: An integrated tensor ``x[:, :]``.
    """
    # xs: List[Tensor]
    xs = []
    x = init.unsqueeze(-2)
    shift = shift.unsqueeze(-3)
    for t in range(trans.size(-3)):
        x = x @ trans[..., t, :, :] + shift[..., t, :]
        xs.append(x)
    return torch.cat(xs, dim=-2)


def _logmatmulexp(x, y):
    """
    Numerically stable version of ``(x.log() @ y.log()).exp()``.
    """
    x_shift = x.max(-1, keepdim=True)[0]
    y_shift = y.max(-2, keepdim=True)[0]
    xy = torch.matmul((x - x_shift).exp(), (y - y_shift).exp()).log()
    return xy + x_shift + y_shift


@torch.jit.script
def _sequential_logmatmulexp(logits):
    """
    For a tensor ``x`` whose time dimension is -3, computes::

        x[..., 0, :, :] @ x[..., 1, :, :] @ ... @ x[..., T-1, :, :]

    but does so numerically stably in log space.
    """
    batch_shape = logits.shape[:-3]
    state_dim = logits.size(-1)
    while logits.size(-3) > 1:
        time = logits.size(-3)
        even_time = time // 2 * 2
        even_part = logits[..., :even_time, :, :]
        x_y = even_part.reshape(batch_shape + (even_time // 2, 2, state_dim, state_dim))
        x, y = x_y.unbind(-3)
        contracted = _logmatmulexp(x, y)
        if time > even_time:
            contracted = torch.cat((contracted, logits[..., -1:, :, :]), dim=-3)
        logits = contracted
    return logits.squeeze(-3)


def _sequential_gaussian_tensordot(gaussian):
    """
    Integrates a Gaussian ``x`` whose rightmost batch dimension is time, computes::

        x[..., 0] @ x[..., 1] @ ... @ x[..., T-1]
    """
    assert isinstance(gaussian, Gaussian)
    assert gaussian.dim() % 2 == 0, "dim is not even"
    batch_shape = gaussian.batch_shape[:-1]
    state_dim = gaussian.dim() // 2
    while gaussian.batch_shape[-1] > 1:
        time = gaussian.batch_shape[-1]
        even_time = time // 2 * 2
        even_part = gaussian[..., :even_time]
        x_y = even_part.reshape(batch_shape + (even_time // 2, 2))
        x, y = x_y[..., 0], x_y[..., 1]
        contracted = gaussian_tensordot(x, y, state_dim)
        if time > even_time:
            contracted = Gaussian.cat((contracted, gaussian[..., -1:]), dim=-1)
        gaussian = contracted
    return gaussian[..., 0]


def _sequential_gamma_gaussian_tensordot(gamma_gaussian):
    """
    Integrates a GammaGaussian ``x`` whose rightmost batch dimension is time, computes::

        x[..., 0] @ x[..., 1] @ ... @ x[..., T-1]
    """
    assert isinstance(gamma_gaussian, GammaGaussian)
    assert gamma_gaussian.dim() % 2 == 0, "dim is not even"
    batch_shape = gamma_gaussian.batch_shape[:-1]
    state_dim = gamma_gaussian.dim() // 2
    while gamma_gaussian.batch_shape[-1] > 1:
        time = gamma_gaussian.batch_shape[-1]
        even_time = time // 2 * 2
        even_part = gamma_gaussian[..., :even_time]
        x_y = even_part.reshape(batch_shape + (even_time // 2, 2))
        x, y = x_y[..., 0], x_y[..., 1]
        contracted = gamma_gaussian_tensordot(x, y, state_dim)
        if time > even_time:
            contracted = GammaGaussian.cat((contracted, gamma_gaussian[..., -1:]), dim=-1)
        gamma_gaussian = contracted
    return gamma_gaussian[..., 0]


class DiscreteHMM(TorchDistribution):
    """
    Hidden Markov Model with discrete latent state and arbitrary observation
    distribution. This uses [1] to parallelize over time, achieving
    O(log(time)) parallel complexity.

    The event_shape of this distribution includes time on the left::

        event_shape = (num_steps,) + observation_dist.event_shape

    This distribution supports any combination of homogeneous/heterogeneous
    time dependency of ``transition_logits`` and ``observation_dist``. However,
    because time is included in this distribution's event_shape, the
    homogeneous+homogeneous case will have a broadcastable event_shape with
    ``num_steps = 1``, allowing :meth:`log_prob` to work with arbitrary length
    data::

        # homogeneous + homogeneous case:
        event_shape = (1,) + observation_dist.event_shape

    **References:**

    [1] Simo Sarkka, Angel F. Garcia-Fernandez (2019)
        "Temporal Parallelization of Bayesian Filters and Smoothers"
        https://arxiv.org/pdf/1905.13002.pdf

    :param ~torch.Tensor initial_logits: A logits tensor for an initial
        categorical distribution over latent states. Should have rightmost size
        ``state_dim`` and be broadcastable to ``batch_shape + (state_dim,)``.
    :param ~torch.Tensor transition_logits: A logits tensor for transition
        conditional distributions between latent states. Should have rightmost
        shape ``(state_dim, state_dim)`` (old, new), and be broadcastable to
        ``batch_shape + (num_steps, state_dim, state_dim)``.
    :param ~torch.distributions.Distribution observation_dist: A conditional
        distribution of observed data conditioned on latent state. The
        ``.batch_shape`` should have rightmost size ``state_dim`` and be
        broadcastable to ``batch_shape + (num_steps, state_dim)``. The
        ``.event_shape`` may be arbitrary.
    """
    arg_constraints = {"initial_logits": constraints.real,
                       "transition_logits": constraints.real}

    def __init__(self, initial_logits, transition_logits, observation_dist, validate_args=None):
        if initial_logits.dim() < 1:
            raise ValueError("expected initial_logits to have at least one dim, "
                             "actual shape = {}".format(initial_logits.shape))
        if transition_logits.dim() < 2:
            raise ValueError("expected transition_logits to have at least two dims, "
                             "actual shape = {}".format(transition_logits.shape))
        if len(observation_dist.batch_shape) < 1:
            raise ValueError("expected observation_dist to have at least one batch dim, "
                             "actual .batch_shape = {}".format(observation_dist.batch_shape))
        shape = broadcast_shape(initial_logits.shape[:-1] + (1,),
                                transition_logits.shape[:-2],
                                observation_dist.batch_shape[:-1])
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + observation_dist.event_shape
        self.initial_logits = initial_logits - initial_logits.logsumexp(-1, True)
        self.transition_logits = transition_logits - transition_logits.logsumexp(-1, True)
        self.observation_dist = observation_dist
        super(DiscreteHMM, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(DiscreteHMM, _instance)
        batch_shape = torch.Size(broadcast_shape(self.batch_shape, batch_shape))
        # We only need to expand one of the inputs, since batch_shape is determined
        # by broadcasting all three. To save computation in _sequential_logmatmulexp(),
        # we expand only initial_logits, which is applied only after the logmatmulexp.
        # This is similar to the ._unbroadcasted_* pattern used elsewhere in distributions.
        new.initial_logits = self.initial_logits.expand(batch_shape + (-1,))
        new.transition_logits = self.transition_logits
        new.observation_dist = self.observation_dist
        super(DiscreteHMM, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self.__dict__.get('_validate_args')
        return new

    def log_prob(self, value):
        # Combine observation and transition factors.
        value = value.unsqueeze(-1 - self.observation_dist.event_dim)
        observation_logits = self.observation_dist.log_prob(value)
        result = self.transition_logits + observation_logits.unsqueeze(-2)

        # Eliminate time dimension.
        result = _sequential_logmatmulexp(result)

        # Combine initial factor.
        result = self.initial_logits + result.logsumexp(-1)

        # Marginalize out final state.
        result = result.logsumexp(-1)
        return result

    def filter(self, value):
        """
        Compute posterior over final state given a sequence of observations.

        :param ~torch.Tensor value: A sequence of observations.
        :return: A posterior distribution
            over latent states at the final time step. ``result.logits`` can
            then be used as ``initial_logits`` in a sequential Pyro model for
            prediction.
        :rtype: ~pyro.distributions.Categorical
        """
        # Combine observation and transition factors.
        value = value.unsqueeze(-1 - self.observation_dist.event_dim)
        observation_logits = self.observation_dist.log_prob(value)
        logp = self.transition_logits + observation_logits.unsqueeze(-2)

        # Eliminate time dimension.
        logp = _sequential_logmatmulexp(logp)

        # Combine initial factor.
        logp = (self.initial_logits.unsqueeze(-1) + logp).logsumexp(-2)

        # Convert to a distribution.
        return Categorical(logits=logp, validate_args=self._validate_args)


class GaussianHMM(TorchDistribution):
    """
    Hidden Markov Model with Gaussians for initial, transition, and observation
    distributions. This adapts [1] to parallelize over time to achieve
    O(log(time)) parallel complexity, however it differs in that it tracks the
    log normalizer to ensure :meth:`log_prob` is differentiable.

    This corresponds to the generative model::

        z = initial_distribution.sample()
        x = []
        for t in range(num_events):
            z = z @ transition_matrix + transition_dist.sample()
            x.append(z @ observation_matrix + observation_dist.sample())

    The event_shape of this distribution includes time on the left::

        event_shape = (num_steps,) + observation_dist.event_shape

    This distribution supports any combination of homogeneous/heterogeneous
    time dependency of ``transition_dist`` and ``observation_dist``. However,
    because time is included in this distribution's event_shape, the
    homogeneous+homogeneous case will have a broadcastable event_shape with
    ``num_steps = 1``, allowing :meth:`log_prob` to work with arbitrary length
    data::

        event_shape = (1, obs_dim)  # homogeneous + homogeneous case

    **References:**

    [1] Simo Sarkka, Angel F. Garcia-Fernandez (2019)
        "Temporal Parallelization of Bayesian Filters and Smoothers"
        https://arxiv.org/pdf/1905.13002.pdf

    :ivar int hidden_dim: The dimension of the hidden state.
    :ivar int obs_dim: The dimension of the observed state.
    :param ~torch.distributions.MultivariateNormal initial_dist: A distribution
        over initial states. This should have batch_shape broadcastable to
        ``self.batch_shape``.  This should have event_shape ``(hidden_dim,)``.
    :param ~torch.Tensor transition_matrix: A linear transformation of hidden
        state. This should have shape broadcastable to
        ``self.batch_shape + (num_steps, hidden_dim, hidden_dim)`` where the
        rightmost dims are ordered ``(old, new)``.
    :param ~torch.distributions.MultivariateNormal transition_dist: A process
        noise distribution. This should have batch_shape broadcastable to
        ``self.batch_shape + (num_steps,)``.  This should have event_shape
        ``(hidden_dim,)``.
    :param ~torch.Tensor observation_matrix: A linear transformation from hidden
        to observed state. This should have shape broadcastable to
        ``self.batch_shape + (num_steps, hidden_dim, obs_dim)``.
    :param observation_dist: An observation noise distribution. This should
        have batch_shape broadcastable to ``self.batch_shape + (num_steps,)``.
        This should have event_shape ``(obs_dim,)``.
    :type observation_dist: ~torch.distributions.MultivariateNormal or
        ~torch.distributions.Independent of ~torch.distributions.Normal
    """
    arg_constraints = {}

    def __init__(self, initial_dist, transition_matrix, transition_dist,
                 observation_matrix, observation_dist, validate_args=None):
        assert (isinstance(initial_dist, torch.distributions.MultivariateNormal) or
                (isinstance(initial_dist, torch.distributions.Independent) and
                 isinstance(initial_dist.base_dist, torch.distributions.Normal)))
        assert isinstance(transition_matrix, torch.Tensor)
        assert (isinstance(transition_dist, torch.distributions.MultivariateNormal) or
                (isinstance(transition_dist, torch.distributions.Independent) and
                 isinstance(transition_dist.base_dist, torch.distributions.Normal)))
        assert isinstance(observation_matrix, torch.Tensor)
        assert (isinstance(observation_dist, torch.distributions.MultivariateNormal) or
                (isinstance(observation_dist, torch.distributions.Independent) and
                 isinstance(observation_dist.base_dist, torch.distributions.Normal)))
        hidden_dim, obs_dim = observation_matrix.shape[-2:]
        assert initial_dist.event_shape == (hidden_dim,)
        assert transition_matrix.shape[-2:] == (hidden_dim, hidden_dim)
        assert transition_dist.event_shape == (hidden_dim,)
        assert observation_dist.event_shape == (obs_dim,)
        shape = broadcast_shape(initial_dist.batch_shape + (1,),
                                transition_matrix.shape[:-2],
                                transition_dist.batch_shape,
                                observation_matrix.shape[:-2],
                                observation_dist.batch_shape)
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + (obs_dim,)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.initial_dist = initial_dist
        self.transition_matrix = transition_matrix
        self.transition_dist = transition_dist
        self.observation_matrix = observation_matrix
        self.observation_dist = observation_dist

    @lazy_property
    def _init(self):
        # To save computation in _sequential_gaussian_tensordot(), we expand
        # only _init, which is applied only after
        # _sequential_gaussian_tensordot().
        return mvn_to_gaussian(self.initial_dist).expand(self.batch_shape)

    @lazy_property
    def _trans(self):
        return matrix_and_mvn_to_gaussian(self.transition_matrix, self.transition_dist)

    @lazy_property
    def _obs(self):
        return matrix_and_mvn_to_gaussian(self.observation_matrix, self.observation_dist)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GaussianHMM, _instance)
        new.hidden_dim = self.hidden_dim
        new.obs_dim = self.obs_dim
        new.initial_dist = self.initial_dist
        new.transition_matrix = self.transition_matrix
        new.transition_dist = self.transition_dist
        new.observation_matrix = self.observation_matrix
        new.observation_dist = self.observation_dist

        # Expand lazily.
        batch_shape = torch.Size(broadcast_shape(self.batch_shape, batch_shape))
        super(GaussianHMM, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self.__dict__.get('_validate_args')
        return new

    def log_prob(self, value):
        # Combine observation and transition factors.
        result = self._trans + self._obs.condition(value).event_pad(left=self.hidden_dim)

        # Eliminate time dimension.
        result = _sequential_gaussian_tensordot(result.expand(result.batch_shape))

        # Combine initial factor.
        result = gaussian_tensordot(self._init, result, dims=self.hidden_dim)

        # Marginalize out final state.
        result = result.event_logsumexp()
        return result

    def rsample(self, sample_shape=torch.Size()):
        batch_shape = self.batch_shape
        time_shape = self.event_shape[:1]
        init = self.initial_dist.expand(batch_shape).rsample(sample_shape)
        trans = self.transition_dist.expand(batch_shape + time_shape).rsample(sample_shape)
        obs = self.observation_dist.expand(batch_shape + time_shape).rsample(sample_shape)
        mat = self.transition_matrix.expand(batch_shape + time_shape + (self.hidden_dim, self.hidden_dim))
        z = _linear_integrate(init, mat, trans)
        return (z.unsqueeze(-2) @ self.observation_matrix).squeeze(-2) + obs

    def filter(self, value):
        """
        Compute posterior over final state given a sequence of observations.

        :param ~torch.Tensor value: A sequence of observations.
        :return: A posterior
            distribution over latent states at the final time step. ``result``
            can then be used as ``initial_dist`` in a sequential Pyro model for
            prediction.
        :rtype: ~pyro.distributions.MultivariateNormal
        """
        # Combine observation and transition factors.
        logp = self._trans + self._obs.condition(value).event_pad(left=self.hidden_dim)

        # Eliminate time dimension.
        logp = _sequential_gaussian_tensordot(logp.expand(logp.batch_shape))

        # Combine initial factor.
        logp = gaussian_tensordot(self._init, logp, dims=self.hidden_dim)

        # Convert to a distribution
        precision = logp.precision
        loc = logp.info_vec.unsqueeze(-1).cholesky_solve(precision.cholesky()).squeeze(-1)
        return MultivariateNormal(loc, precision_matrix=precision,
                                  validate_args=self._validate_args)


class GammaGaussianHMM(TorchDistribution):
    """
    Hidden Markov Model with the joint distribution of initial state, hidden
    state, and observed state is a :class:`~pyro.distributions.MultivariateStudentT`
    distribution along the line of references [2] and [3]. This adapts [1]
    to parallelize over time to achieve O(log(time)) parallel complexity.

    This GammaGaussianHMM class corresponds to the generative model::

        s = Gamma(df/2, df/2).sample()
        z = scale(initial_dist, s).sample()
        x = []
        for t in range(num_events):
            z = z @ transition_matrix + scale(transition_dist, s).sample()
            x.append(z @ observation_matrix + scale(observation_dist, s).sample())

    where `scale(mvn(loc, precision), s) := mvn(loc, s * precision)`.

    The event_shape of this distribution includes time on the left::

        event_shape = (num_steps,) + observation_dist.event_shape

    This distribution supports any combination of homogeneous/heterogeneous
    time dependency of ``transition_dist`` and ``observation_dist``. However,
    because time is included in this distribution's event_shape, the
    homogeneous+homogeneous case will have a broadcastable event_shape with
    ``num_steps = 1``, allowing :meth:`log_prob` to work with arbitrary length
    data::

        event_shape = (1, obs_dim)  # homogeneous + homogeneous case

    **References:**

    [1] Simo Sarkka, Angel F. Garcia-Fernandez (2019)
        "Temporal Parallelization of Bayesian Filters and Smoothers"
        https://arxiv.org/pdf/1905.13002.pdf

    [2] F. J. Giron and J. C. Rojano (1994)
        "Bayesian Kalman filtering with elliptically contoured errors"

    [3] Filip Tronarp, Toni Karvonen, and Simo Sarkka (2019)
        "Student's t-filters for noise scale estimation"
        https://users.aalto.fi/~ssarkka/pub/SPL2019.pdf

    :ivar int hidden_dim: The dimension of the hidden state.
    :ivar int obs_dim: The dimension of the observed state.
    :param Gamma scale_dist: Prior of the mixing distribution.
    :param MultivariateNormal initial_dist: A distribution with unit scale mixing
        over initial states. This should have batch_shape broadcastable to
        ``self.batch_shape``.  This should have event_shape ``(hidden_dim,)``.
    :param ~torch.Tensor transition_matrix: A linear transformation of hidden
        state. This should have shape broadcastable to
        ``self.batch_shape + (num_steps, hidden_dim, hidden_dim)`` where the
        rightmost dims are ordered ``(old, new)``.
    :param MultivariateNormal transition_dist: A process noise distribution
        with unit scale mixing. This should have batch_shape broadcastable to
        ``self.batch_shape + (num_steps,)``. This should have event_shape
        ``(hidden_dim,)``.
    :param ~torch.Tensor observation_matrix: A linear transformation from hidden
        to observed state. This should have shape broadcastable to
        ``self.batch_shape + (num_steps, hidden_dim, obs_dim)``.
    :param MultivariateNormal observation_dist: An observation noise distribution
        with unit scale mixing. This should have batch_shape broadcastable to
        ``self.batch_shape + (num_steps,)``.
        This should have event_shape ``(obs_dim,)``.
    """
    arg_constraints = {}

    def __init__(self, scale_dist, initial_dist, transition_matrix, transition_dist,
                 observation_matrix, observation_dist, validate_args=None):
        assert isinstance(scale_dist, Gamma)
        assert isinstance(initial_dist, MultivariateNormal)
        assert isinstance(transition_matrix, torch.Tensor)
        assert isinstance(transition_dist, MultivariateNormal)
        assert isinstance(observation_matrix, torch.Tensor)
        assert isinstance(observation_dist, MultivariateNormal)
        hidden_dim, obs_dim = observation_matrix.shape[-2:]
        assert initial_dist.event_shape == (hidden_dim,)
        assert transition_matrix.shape[-2:] == (hidden_dim, hidden_dim)
        assert transition_dist.event_shape == (hidden_dim,)
        assert observation_dist.event_shape == (obs_dim,)
        shape = broadcast_shape(scale_dist.batch_shape + (1,),
                                initial_dist.batch_shape + (1,),
                                transition_matrix.shape[:-2],
                                transition_dist.batch_shape,
                                observation_matrix.shape[:-2],
                                observation_dist.batch_shape)
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + (obs_dim,)
        super(GammaGaussianHMM, self).__init__(batch_shape, event_shape, validate_args=validate_args)
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self._init = gamma_and_mvn_to_gamma_gaussian(scale_dist, initial_dist)
        self._trans = matrix_and_mvn_to_gamma_gaussian(transition_matrix, transition_dist)
        self._obs = matrix_and_mvn_to_gamma_gaussian(observation_matrix, observation_dist)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GammaGaussianHMM, _instance)
        batch_shape = torch.Size(broadcast_shape(self.batch_shape, batch_shape))
        new.hidden_dim = self.hidden_dim
        new.obs_dim = self.obs_dim
        # We only need to expand one of the inputs, since batch_shape is determined
        # by broadcasting all three. To save computation in _sequential_gaussian_tensordot(),
        # we expand only _init, which is applied only after _sequential_gaussian_tensordot().
        new._init = self._init.expand(batch_shape)
        new._trans = self._trans
        new._obs = self._obs
        super(GammaGaussianHMM, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self.__dict__.get('_validate_args')
        return new

    def log_prob(self, value):
        # Combine observation and transition factors.
        result = self._trans + self._obs.condition(value).event_pad(left=self.hidden_dim)

        # Eliminate time dimension.
        result = _sequential_gamma_gaussian_tensordot(result.expand(result.batch_shape))

        # Combine initial factor.
        result = gamma_gaussian_tensordot(self._init, result, dims=self.hidden_dim)

        # Marginalize out final state.
        result = result.event_logsumexp()

        # Marginalize out multiplier.
        result = result.logsumexp()
        return result

    def filter(self, value):
        """
        Compute posteriors over the multiplier and the final state
        given a sequence of observations. The posterior is a pair of
        Gamma and MultivariateNormal distributions (i.e. a GammaGaussian
        instance).

        :param ~torch.Tensor value: A sequence of observations.
        :return: A pair of posterior distributions over the mixing and the latent
            state at the final time step.
        :rtype: a tuple of ~pyro.distributions.Gamma and ~pyro.distributions.MultivariateNormal
        """
        # Combine observation and transition factors.
        logp = self._trans + self._obs.condition(value).event_pad(left=self.hidden_dim)

        # Eliminate time dimension.
        logp = _sequential_gamma_gaussian_tensordot(logp.expand(logp.batch_shape))

        # Combine initial factor.
        logp = gamma_gaussian_tensordot(self._init, logp, dims=self.hidden_dim)

        # Posterior of the scale
        gamma_dist = logp.event_logsumexp()
        scale_post = Gamma(gamma_dist.concentration, gamma_dist.rate,
                           validate_args=self._validate_args)
        # Conditional of last state on unit scale
        scale_tril = logp.precision.cholesky()
        loc = logp.info_vec.unsqueeze(-1).cholesky_solve(scale_tril).squeeze(-1)
        mvn = MultivariateNormal(loc, scale_tril=scale_tril,
                                 validate_args=self._validate_args)
        return scale_post, mvn


class LinearHMM(TorchDistribution):
    r"""
    Hidden Markov Model with linear dynamics and observations and arbitrary
    noise for initial, transition, and observation distributions.  Each of
    those distributions can be e.g.
    `:class:`~pyro.distributions.MultivariateNormal` or
    `:class:`~pyro.distributions.Independent` of
    `:class:`~pyro.distributions.Normal`,
    `:class:`~pyro.distributions.StudentT`, or `~pyro.distributions.Stable` .
    Additionally the observation distribution may be constrained, e.g.
    `~pyro.distributions.LogNormal`

    This corresponds to the generative model::

        z = initial_distribution.sample()
        x = []
        for t in range(num_events):
            z = z @ transition_matrix + transition_dist.sample()
            y = z @ observation_matrix + obs_base_dist.sample()
            x.append(obs_transform(y))

    where ``observation_dist`` is split into ``obs_base_dist`` and an optional
    ``obs_transform`` (defaulting to the identity).

    This implements a reparameterized :meth:`rsample` method but does not
    implement a :meth:`log_prob` method. Derived classes may implement
    :meth:`log_prob` .

    Inference without :meth:`log_prob` can be performed using either
    reparameterization with :class:`~pyro.infer.reparam.hmm.LinearHMMReparam`
    or likelihood-free algorithms such as
    :class:`~pyro.infer.energy_distance.EnergyDistance` .  Note that while
    stable processes generally require a common shared stability parameter
    :math:`\alpha` , this distribution and the above inference algorithms allow
    heterogeneous stability parameters.

    The event_shape of this distribution includes time on the left::

        event_shape = (num_steps,) + observation_dist.event_shape

    This distribution supports any combination of homogeneous/heterogeneous
    time dependency of ``transition_dist`` and ``observation_dist``. However at
    least one of the distributions or matrices must be expanded to contain the
    time dimension.

    :ivar int hidden_dim: The dimension of the hidden state.
    :ivar int obs_dim: The dimension of the observed state.
    :param initial_dist: A distribution over initial states. This should have
        batch_shape broadcastable to ``self.batch_shape``.  This should have
        event_shape ``(hidden_dim,)``.
    :param ~torch.Tensor transition_matrix: A linear transformation of hidden
        state. This should have shape broadcastable to ``self.batch_shape +
        (num_steps, hidden_dim, hidden_dim)`` where the rightmost dims are
        ordered ``(old, new)``.
    :param transition_dist: A distribution over process noise. This should have
        batch_shape broadcastable to ``self.batch_shape + (num_steps,)``.  This
        should have event_shape ``(hidden_dim,)``.
    :param ~torch.Tensor observation_matrix: A linear transformation from hidden
        to observed state. This should have shape broadcastable to
        ``self.batch_shape + (num_steps, hidden_dim, obs_dim)``.
    :param observation_dist: A observation noise distribution. This should have
        batch_shape broadcastable to ``self.batch_shape + (num_steps,)``.  This
        should have event_shape ``(obs_dim,)``.
    """
    arg_constraints = {}
    has_rsample = True

    def __init__(self, initial_dist, transition_matrix, transition_dist,
                 observation_matrix, observation_dist, validate_args=None):
        assert initial_dist.has_rsample
        assert initial_dist.event_dim == 1
        assert (isinstance(transition_matrix, torch.Tensor) and
                transition_matrix.dim() >= 2)
        assert transition_dist.has_rsample
        assert transition_dist.event_dim == 1
        assert (isinstance(observation_matrix, torch.Tensor) and
                observation_matrix.dim() >= 2)
        assert observation_dist.has_rsample
        assert observation_dist.event_dim == 1

        hidden_dim, obs_dim = observation_matrix.shape[-2:]
        assert initial_dist.event_shape == (hidden_dim,)
        assert transition_matrix.shape[-2:] == (hidden_dim, hidden_dim)
        assert transition_dist.event_shape == (hidden_dim,)
        assert observation_dist.event_shape == (obs_dim,)
        shape = broadcast_shape(initial_dist.batch_shape + (1,),
                                transition_matrix.shape[:-2],
                                transition_dist.batch_shape,
                                observation_matrix.shape[:-2],
                                observation_dist.batch_shape)
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + (obs_dim,)
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

        # Expand eagerly.
        if initial_dist.batch_shape != batch_shape:
            initial_dist = initial_dist.expand(batch_shape)
        if transition_matrix.shape[:-2] != batch_shape + time_shape:
            transition_matrix = transition_matrix.expand(
                batch_shape + time_shape + (hidden_dim, hidden_dim))
        if transition_dist.batch_shape != batch_shape + time_shape:
            transition_dist = transition_dist.expand(batch_shape + time_shape)
        if observation_matrix.shape[:-2] != batch_shape + time_shape:
            observation_matrix = observation_matrix.expand(
                batch_shape + time_shape + (hidden_dim, obs_dim))
        if observation_dist.batch_shape != batch_shape + time_shape:
            observation_dist = observation_dist.expand(batch_shape + time_shape)

        # Extract observation transforms.
        transforms = []
        while True:
            if isinstance(observation_dist, torch.distributions.Independent):
                observation_dist = observation_dist.base_dist
            elif isinstance(observation_dist, torch.distributions.TransformedDistribution):
                transforms = observation_dist.transforms + transforms
                observation_dist = observation_dist.base_dist
            else:
                break
        if not observation_dist.event_shape:
            observation_dist = Independent(observation_dist, 1)

        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.initial_dist = initial_dist
        self.transition_matrix = transition_matrix
        self.transition_dist = transition_dist
        self.observation_matrix = observation_matrix
        self.observation_dist = observation_dist
        self.transforms = transforms

    @property
    def support(self):
        return self.observation_dist.support

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(LinearHMM, _instance)
        batch_shape = torch.Size(batch_shape)
        time_shape = self.transition_dist.batch_shape[-1:]
        new.hidden_dim = self.hidden_dim
        new.obs_dim = self.obs_dim
        new.initial_dist = self.initial_dist.expand(batch_shape)
        new.transition_matrix = self.transition_matrix.expand(
            batch_shape + time_shape + (self.hidden_dim, self.hidden_dim))
        new.transition_dist = self.transition_dist.expand(batch_shape + time_shape)
        new.observation_matrix = self.observation_matrix.expand(
            batch_shape + time_shape + (self.hidden_dim, self.obs_dim))
        new.observation_dist = self.observation_dist.expand(batch_shape + time_shape)
        new.transforms = self.transforms
        super(LinearHMM, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self.__dict__.get('_validate_args')
        return new

    def log_prob(self, value):
        raise NotImplementedError("LinearHMM.log_prob() is not implemented")

    def rsample(self, sample_shape=torch.Size()):
        init = self.initial_dist.rsample(sample_shape)
        trans = self.transition_dist.rsample(sample_shape)
        obs = self.observation_dist.rsample(sample_shape)
        z = _linear_integrate(init, self.transition_matrix, trans)
        x = (z.unsqueeze(-2) @ self.observation_matrix).squeeze(-2) + obs
        for t in self.transforms:
            x = t(x)
        return x


class GaussianMRF(TorchDistribution):
    """
    Temporal Markov Random Field with Gaussian factors for initial, transition,
    and observation distributions. This adapts [1] to parallelize over time to
    achieve O(log(time)) parallel complexity, however it differs in that it
    tracks the log normalizer to ensure :meth:`log_prob` is differentiable.

    The event_shape of this distribution includes time on the left::

        event_shape = (num_steps,) + observation_dist.event_shape

    This distribution supports any combination of homogeneous/heterogeneous
    time dependency of ``transition_dist`` and ``observation_dist``. However,
    because time is included in this distribution's event_shape, the
    homogeneous+homogeneous case will have a broadcastable event_shape with
    ``num_steps = 1``, allowing :meth:`log_prob` to work with arbitrary length
    data::

        event_shape = (1, obs_dim)  # homogeneous + homogeneous case

    **References:**

    [1] Simo Sarkka, Angel F. Garcia-Fernandez (2019)
        "Temporal Parallelization of Bayesian Filters and Smoothers"
        https://arxiv.org/pdf/1905.13002.pdf

    :ivar int hidden_dim: The dimension of the hidden state.
    :ivar int obs_dim: The dimension of the observed state.
    :param ~torch.distributions.MultivariateNormal initial_dist: A distribution
        over initial states. This should have batch_shape broadcastable to
        ``self.batch_shape``.  This should have event_shape ``(hidden_dim,)``.
    :param ~torch.distributions.MultivariateNormal transition_dist: A joint
        distribution factor over a pair of successive time steps. This should
        have batch_shape broadcastable to ``self.batch_shape + (num_steps,)``.
        This should have event_shape ``(hidden_dim + hidden_dim,)`` (old+new).
    :param ~torch.distributions.MultivariateNormal observation_dist: A joint
        distribution factor over a hidden and an observed state. This should
        have batch_shape broadcastable to ``self.batch_shape + (num_steps,)``.
        This should have event_shape ``(hidden_dim + obs_dim,)``.
    """
    arg_constraints = {}

    def __init__(self, initial_dist, transition_dist, observation_dist, validate_args=None):
        assert isinstance(initial_dist, torch.distributions.MultivariateNormal)
        assert isinstance(transition_dist, torch.distributions.MultivariateNormal)
        assert isinstance(observation_dist, torch.distributions.MultivariateNormal)
        hidden_dim = initial_dist.event_shape[0]
        assert transition_dist.event_shape[0] == hidden_dim + hidden_dim
        obs_dim = observation_dist.event_shape[0] - hidden_dim

        shape = broadcast_shape(initial_dist.batch_shape + (1,),
                                transition_dist.batch_shape,
                                observation_dist.batch_shape)
        batch_shape, time_shape = shape[:-1], shape[-1:]
        event_shape = time_shape + (obs_dim,)
        super(GaussianMRF, self).__init__(batch_shape, event_shape, validate_args=validate_args)
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self._init = mvn_to_gaussian(initial_dist)
        self._trans = mvn_to_gaussian(transition_dist)
        self._obs = mvn_to_gaussian(observation_dist)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GaussianMRF, _instance)
        batch_shape = torch.Size(broadcast_shape(self.batch_shape, batch_shape))
        new.hidden_dim = self.hidden_dim
        new.obs_dim = self.obs_dim
        # We only need to expand one of the inputs, since batch_shape is determined
        # by broadcasting all three. To save computation in _sequential_gaussian_tensordot(),
        # we expand only _init, which is applied only after _sequential_gaussian_tensordot().
        new._init = self._init.expand(batch_shape)
        new._trans = self._trans
        new._obs = self._obs
        super(GaussianMRF, new).__init__(batch_shape, self.event_shape, validate_args=False)
        new._validate_args = self.__dict__.get('_validate_args')
        return new

    def log_prob(self, value):
        # We compute a normalized distribution as p(obs,hidden) / p(hidden).
        logp_oh = self._trans
        logp_h = self._trans

        # Combine observation and transition factors.
        logp_oh += self._obs.condition(value).event_pad(left=self.hidden_dim)
        logp_h += self._obs.marginalize(right=self.obs_dim).event_pad(left=self.hidden_dim)

        # Concatenate p(obs,hidden) and p(hidden) into a single Gaussian.
        batch_dim = 1 + max(len(self._init.batch_shape) + 1, len(logp_oh.batch_shape))
        batch_shape = (1,) * (batch_dim - len(logp_oh.batch_shape)) + logp_oh.batch_shape
        logp = Gaussian.cat([logp_oh.expand(batch_shape),
                             logp_h.expand(batch_shape)])

        # Eliminate time dimension.
        logp = _sequential_gaussian_tensordot(logp)

        # Combine initial factor.
        logp = gaussian_tensordot(self._init, logp, dims=self.hidden_dim)

        # Marginalize out final state.
        logp_oh, logp_h = logp.event_logsumexp()
        return logp_oh - logp_h  # = log( p(obs,hidden) / p(hidden) )
