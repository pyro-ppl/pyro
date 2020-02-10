# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
import torch
from collections import namedtuple

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
                 is_diag_mass=True):
        self.adapt_step_size = adapt_step_size
        self.adapt_mass_matrix = adapt_mass_matrix
        self.target_accept_prob = target_accept_prob
        self.is_diag_mass = is_diag_mass
        self.step_size = 1 if step_size is None else step_size
        self._init_step_size = self.step_size
        self._adaptation_disabled = not (adapt_step_size or adapt_mass_matrix)
        if adapt_step_size:
            self._step_size_adapt_scheme = DualAveraging()
        if adapt_mass_matrix:
            self._mass_matrix_adapt_scheme = WelfordCovariance(diagonal=is_diag_mass)

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
        self._inverse_mass_matrix = None
        self._r_dist = None
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
        loc = torch.zeros(self._inverse_mass_matrix.size(0),
                          dtype=self._inverse_mass_matrix.dtype,
                          device=self._inverse_mass_matrix.device)
        if self.is_diag_mass:
            self._r_dist = dist.Normal(loc, self._inverse_mass_matrix.rsqrt())
        else:
            self._r_dist = dist.MultivariateNormal(loc,
                                                   precision_matrix=self._inverse_mass_matrix)

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
            self._mass_matrix_adapt_scheme.reset()

    def step(self, t, z, accept_prob):
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
            z_flat = torch.cat([z[name].reshape(-1) for name in sorted(z)])
            self._mass_matrix_adapt_scheme.update(z_flat.detach())
        if t == window.end:
            if self._current_window == num_windows - 1:
                self._current_window += 1
                self._end_adaptation()
                return

            if self._current_window == 0:
                self._current_window += 1
                return

            if mass_matrix_adaptation_phase:
                self.inverse_mass_matrix = self._mass_matrix_adapt_scheme.get_covariance()
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
        self._inverse_mass_matrix = value
        self._update_r_dist()
        if self.adapt_mass_matrix:
            self._mass_matrix_adapt_scheme.reset()

    @property
    def r_dist(self):
        return self._r_dist
