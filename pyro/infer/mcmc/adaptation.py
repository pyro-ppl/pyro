# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import OrderedDict, namedtuple

import torch

import pyro
import pyro.distributions as dist
from pyro.ops.dual_averaging import DualAveraging
from pyro.ops.welford import WelfordCovariance

adapt_window = namedtuple("adapt_window", ["start", "end"])


class WarmupAdapter:
    r"""
    Adapts tunable parameters, namely step size and mass matrix, during the
    warmup phase. This class provides lookup properties to read the latest
    values of ``step_size`` and ``inverse_mass_matrix``. These values are
    periodically updated when adaptation is engaged.
    """

    def __init__(self,
                 step_size=1,
                 adapt_step_size=False,
                 target_accept_prob=0.8,
                 adapt_mass_matrix=False,
                 dense_mass=False):
        self.adapt_step_size = adapt_step_size
        self.adapt_mass_matrix = adapt_mass_matrix
        self.target_accept_prob = target_accept_prob
        self.dense_mass = dense_mass
        self.step_size = 1 if step_size is None else step_size
        self._init_step_size = self.step_size
        self._adaptation_disabled = not (adapt_step_size or adapt_mass_matrix)
        if adapt_step_size:
            self._step_size_adapt_scheme = DualAveraging()
        if adapt_mass_matrix:
            self._mass_matrix_adapt_scheme = {}

        # We separate warmup_steps into windows:
        #   start_buffer + window 1 + window 2 + window 3 + ... + end_buffer
        # where the length of each window will be doubled for the next window.
        # We won't adapt mass matrix during start and end buffers; and mass
        # matrix will be updated at the end of each window. This is helpful
        # for dealing with the intense computation of sampling momentum from the
        # inverse of mass matrix.
        self._adapt_start_buffer = 75  # from Stan
        self._adapt_end_buffer = 50  # from Stan
        self._adapt_initial_window = 25  # from Stan

        # configured later on setup
        self._warmup_steps = None
        self._inverse_mass_matrix = OrderedDict()
        self._r_dist = {}
        self._adaptation_schedule = []

    def _build_adaptation_schedule(self):
        adaptation_schedule = []
        # from Stan, for small warmup_steps < 20
        if self._warmup_steps < 20:
            adaptation_schedule.append(adapt_window(0, self._warmup_steps - 1))
            return adaptation_schedule

        start_buffer_size = self._adapt_start_buffer
        end_buffer_size = self._adapt_end_buffer
        init_window_size = self._adapt_initial_window
        if (self._adapt_start_buffer + self._adapt_end_buffer
                + self._adapt_initial_window > self._warmup_steps):
            start_buffer_size = int(0.15 * self._warmup_steps)
            end_buffer_size = int(0.1 * self._warmup_steps)
            init_window_size = self._warmup_steps - start_buffer_size - end_buffer_size
        adaptation_schedule.append(adapt_window(start=0, end=start_buffer_size - 1))
        end_window_start = self._warmup_steps - end_buffer_size

        next_window_size = init_window_size
        next_window_start = start_buffer_size
        while next_window_start < end_window_start:
            cur_window_start, cur_window_size = next_window_start, next_window_size
            # Ensure that slow adaptation windows are monotonically increasing
            if 3 * cur_window_size <= end_window_start - cur_window_start:
                next_window_size = 2 * cur_window_size
            else:
                cur_window_size = end_window_start - cur_window_start
            next_window_start = cur_window_start + cur_window_size
            adaptation_schedule.append(adapt_window(cur_window_start, next_window_start - 1))
        adaptation_schedule.append(adapt_window(end_window_start,
                                                self._warmup_steps - 1))
        return adaptation_schedule

    def reset_step_size_adaptation(self, z):
        r"""
        Finds a reasonable step size and resets step size adaptation scheme.
        """
        if self._find_reasonable_step_size is not None:
            with pyro.validation_enabled(False):
                self.step_size = self._find_reasonable_step_size(z)
        self._step_size_adapt_scheme.prox_center = math.log(10 * self.step_size)
        self._step_size_adapt_scheme.reset()

    def _update_step_size(self, accept_prob):
        # calculate a statistic for Dual Averaging scheme
        H = self.target_accept_prob - accept_prob
        self._step_size_adapt_scheme.step(H)
        log_step_size, _ = self._step_size_adapt_scheme.get_state()
        self.step_size = math.exp(log_step_size)

    def _update_r_dist(self):
        for site_names, inv_mass_matrix in self._inverse_mass_matrix.items():
            loc = inv_mass_matrix.new_zeros(inv_mass_matrix.size(0))
            if inv_mass_matrix.dim() == 1:
                self._r_dist[site_names] = dist.Normal(loc, inv_mass_matrix.rsqrt())
            else:
                self._r_dist[site_names] = dist.MultivariateNormal(loc, precision_matrix=inv_mass_matrix)

    def _end_adaptation(self):
        if self.adapt_step_size:
            _, log_step_size_avg = self._step_size_adapt_scheme.get_state()
            self.step_size = math.exp(log_step_size_avg)

    def configure(self, warmup_steps, initial_step_size=None, inv_mass_matrix=None,
                  find_reasonable_step_size_fn=None):
        r"""
        Model specific properties that are specified when the HMC kernel is setup.

        :param warmup_steps: Number of warmup steps that the sampler is initialized with.
        :param initial_step_size: Step size to use to initialize the Dual Averaging scheme.
        :param inv_mass_matrix: Initial value of the inverse mass matrix.
        :param find_reasonable_step_size_fn: A callable to find reasonable step size when
            mass matrix is changed.
        """
        self._warmup_steps = warmup_steps
        self.step_size = initial_step_size if initial_step_size is not None else self._init_step_size
        if find_reasonable_step_size_fn is not None:
            self._find_reasonable_step_size = find_reasonable_step_size_fn
        self.inverse_mass_matrix = inv_mass_matrix
        if self.inverse_mass_matrix is None or self.step_size is None:
            raise ValueError("Incomplete configuration - step size and inverse mass matrix "
                             "need to be initialized.")
        if not self._adaptation_disabled:
            self._adaptation_schedule = self._build_adaptation_schedule()
        self._current_window = 0  # starting window index
        if self.adapt_step_size:
            self._step_size_adapt_scheme.reset()
        if self.adapt_mass_matrix:
            for site_names, inv_mass_matrix in self._inverse_mass_matrix.items():
                adapt_scheme = WelfordCovariance(diagonal=inv_mass_matrix.dim() == 1)
                adapt_scheme.reset()
                self._mass_matrix_adapt_scheme[site_names] = adapt_scheme

    def step(self, t, z, accept_prob, z_grads=None):
        r"""
        Called at each step during the warmup phase to learn tunable
        parameters.

        :param int t: time step, beginning at 0.
        :param dict z: latent variables.
        :param float accept_prob: acceptance probability of the proposal.
        """
        if t >= self._warmup_steps or self._adaptation_disabled:
            return
        window = self._adaptation_schedule[self._current_window]
        num_windows = len(self._adaptation_schedule)
        mass_matrix_adaptation_phase = self.adapt_mass_matrix and \
            (0 < self._current_window < num_windows - 1)
        if self.adapt_step_size:
            self._update_step_size(accept_prob.item())
        if mass_matrix_adaptation_phase:
            for site_names, adapt_scheme in self._mass_matrix_adapt_scheme.items():
                # z_flat = torch.cat([z[name].reshape(-1) for name in site_names])
                # adapt_scheme.update(z_flat.detach())
                z_grads_flat = torch.cat([z_grads[name].reshape(-1) for name in site_names])
                adapt_scheme.update(z_grads_flat)

        if t == window.end:
            if self._current_window == num_windows - 1:
                self._current_window += 1
                self._end_adaptation()
                return

            if self._current_window == 0:
                self._current_window += 1
                return

            if mass_matrix_adaptation_phase:
                inv_mass_matrix = {}
                for site_names, scheme in self._mass_matrix_adapt_scheme.items():
                    prec = scheme.get_covariance(regularize=False)
                    inv_mass_matrix[site_names] = prec.inverse() if prec.ndim == 2 else prec.reciprocal()
                self.inverse_mass_matrix = inv_mass_matrix
                if self.adapt_step_size:
                    self.reset_step_size_adaptation(z)

            self._current_window += 1

    @property
    def adaptation_schedule(self):
        return self._adaptation_schedule

    @property
    def inverse_mass_matrix(self):
        return self._inverse_mass_matrix

    @inverse_mass_matrix.setter
    def inverse_mass_matrix(self, value):
        for site_names, inv_mass_matrix in value.items():
            self._inverse_mass_matrix[site_names] = inv_mass_matrix
        self._update_r_dist()
        if self.adapt_mass_matrix:
            for site_names, adapt_scheme in self._mass_matrix_adapt_scheme.items():
                adapt_scheme.reset()

    @property
    def r_dist(self):
        return self._r_dist


class BlockMassMatrix:
    def __init__(self):
        self._adapt_scheme = {}
        self._inverse_mass_matrix = OrderedDict()
        self._r_scale = {}

    def configure(self, inverse_mass_matrix):
        self.inverse_mass_matrix = inverse_mass_matrix
        for site_names, inv_mass_matrix in inverse_mass_matrix.items():
            head_size = 0 if inv_mass_matrix.dim() == 1 else inv_mass_matrix.size(0)
            adapt_scheme = WelfordArrowheadCovariance(head_size=head_size)
            adapt_scheme.reset()
            self._adapt_scheme[site_names] = adapt_scheme

    def update(self, z, z_grads=None):
        for site_names, adapt_scheme in self._adapt_scheme.items():
            z_flat = torch.cat([z[name].reshape(-1) for name in site_names])
            adapt_scheme.update(z_flat.detach())

    def __call__(self, *args, **kwargs):
        return samples

    def quadratic(self, r1, r2):
        # compute r1 M^{-1} r2
        return self._r_scale.matmul(r1)

    @property
    def inverse_mass_matrix(self):
        return self._inverse_mass_matrix

    @inverse_mass_matrix.setter
    def inverse_mass_matrix(self, value):
        for site_names, inv_mass_matrix in value.items():
            self._inverse_mass_matrix[site_names] = inv_mass_matrix
            # also update r_scale
            self._r_scale = inv_mass_matrix.cholesky()


class ArrowheadMass(InverseMassMatrixAdapter):


    def 


class ArrowheadMassMatrix():
    """
    Computes scale_triu U matrix from a symmetric arrowhead positive definite matrix M,
    that is `M = U @ U.T`. 
    """
    def __init__(self, inverse_mass_matrix):
        self.inverse_mass_matrix

    def __call__(self):
        return 

    N = x.size(-1)
    assert x.dim() == 2
    assert x.size(-2) == N
    assert head_size <= N
    if head_size == 0:
        return x.diag().rsqrt().diag()

    A, B = x[:head_size, :head_size], x[:head_size, head_size:]
    D = x.view(-1)[::N + 1][head_size:]
    # NB: the complexity is O(N * head_size^2)
    # ref: https://en.wikipedia.org/wiki/Schur_complement#Background
    B_Dinv = B / D.unsqueeze(-2)
    schur_complement = A - B_Dinv.matmul(B.transpose(-2, -1))
    tril_upper = precision_to_scale_tril(schur_complement)
    tril_lower_left = -B_Dinv.transpose(-2, -1).matmul(tril_upper)
    # NB: we can exploit this diagonal form of scale_tril
    # to generate samples with O(N x head_size) cost
    tril_lower_diag = D.rsqrt().diag()
    tril_upper = torch.nn.functional.pad(tril_upper, (0, N - head_size))
    tril_lower = torch.cat([tril_lower_left, tril_lower_diag], -1)
    return torch.cat([tril_upper, tril_lower])
