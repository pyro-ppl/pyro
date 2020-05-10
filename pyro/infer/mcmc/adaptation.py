# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import namedtuple

import torch

import pyro
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
        self._mass_matrix_adapter = BlockMassMatrix()

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

    def _end_adaptation(self):
        if self.adapt_step_size:
            _, log_step_size_avg = self._step_size_adapt_scheme.get_state()
            self.step_size = math.exp(log_step_size_avg)

    def configure(self, warmup_steps, initial_step_size=None, mass_matrix_shape=None,
                  find_reasonable_step_size_fn=None, options={}):
        r"""
        Model specific properties that are specified when the HMC kernel is setup.

        :param warmup_steps: Number of warmup steps that the sampler is initialized with.
        :param initial_step_size: Step size to use to initialize the Dual Averaging scheme.
        :param mass_matrix_shape: Shape of the mass matrix.
        :param find_reasonable_step_size_fn: A callable to find reasonable step size when
            mass matrix is changed.
        :param dict options: A dict which maps `dtype`, `device` to the corresponding default
            tensor options. This is used to construct initial mass matrix in `mass_matrix_adapter`.
        """
        self._warmup_steps = warmup_steps
        self.step_size = initial_step_size if initial_step_size is not None else self._init_step_size
        if find_reasonable_step_size_fn is not None:
            self._find_reasonable_step_size = find_reasonable_step_size_fn
        if mass_matrix_shape is None or self.step_size is None:
            raise ValueError("Incomplete configuration - step size and inverse mass matrix "
                             "need to be initialized.")
        self.mass_matrix_adapter.configure(mass_matrix_shape, self.adapt_mass_matrix, options=options)
        if not self._adaptation_disabled:
            self._adaptation_schedule = self._build_adaptation_schedule()
        self._current_window = 0  # starting window index
        if self.adapt_step_size:
            self._step_size_adapt_scheme.reset()

    def step(self, t, z, accept_prob, z_grad=None):
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
            self.mass_matrix_adapter.update(z, z_grad)

        if t == window.end:
            if self._current_window == num_windows - 1:
                self._current_window += 1
                self._end_adaptation()
                return

            if self._current_window == 0:
                self._current_window += 1
                return

            if mass_matrix_adaptation_phase:
                self.mass_matrix_adapter.end_adaptation()
                if self.adapt_step_size:
                    self.reset_step_size_adaptation(z)

            self._current_window += 1

    @property
    def adaptation_schedule(self):
        return self._adaptation_schedule

    @property
    def mass_matrix_adapter(self):
        return self._mass_matrix_adapter

    @mass_matrix_adapter.setter
    def mass_matrix_adapter(self, value):
        self._mass_matrix_adapter = value


# this works for diagonal matrix `x`
def _matmul(x, y):
    return x.mul(y) if x.dim() == 1 else x.matmul(y)


def _cholesky(x):
    return x.sqrt() if x.dim() == 1 else x.cholesky()


def _transpose(x):
    return x if x.dim() == 1 else x.t()


def _upper_triangular_inverse(x):
    if x.dim() == 1:
        return x.reciprocal()
    else:
        identity = torch.eye(x.size(-1), dtype=x.dtype, device=x.device)
        return torch.triangular_solve(identity, x, upper=True)[0]


class BlockMassMatrix:
    """
    This class is used to adapt (inverse) mass matrix and provide useful
    methods to calculate algebraic terms which involves the mass matrix.

    The mass matrix will have block structure, which can be specified by
    using the method :meth:`configure` with the corresponding structured
    `mass_matrix_shape` arg.

    :param float init_scale: initial scale to construct the initial mass matrix.
    """
    def __init__(self, init_scale=1.):
        self._init_scale = init_scale
        self._adapt_scheme = {}
        self._inverse_mass_matrix = {}
        # NB: those sqrt matrices are upper triangular
        self._mass_matrix_sqrt = {}
        self._mass_matrix_sqrt_inverse = {}
        self._mass_matrix_size = {}

    @property
    def mass_matrix_size(self):
        """
        A dict that maps site names to the size of the corresponding mass matrix.
        """
        return self._mass_matrix_size

    @property
    def inverse_mass_matrix(self):
        return self._inverse_mass_matrix

    @inverse_mass_matrix.setter
    def inverse_mass_matrix(self, value):
        for site_names, inverse_mass_matrix in value.items():
            if site_names in self._adapt_scheme:
                self._adapt_scheme[site_names].reset()
            mass_matrix_sqrt_inverse = _transpose(_cholesky(inverse_mass_matrix))
            mass_matrix_sqrt = _upper_triangular_inverse(mass_matrix_sqrt_inverse)
            self._inverse_mass_matrix[site_names] = inverse_mass_matrix
            self._mass_matrix_sqrt[site_names] = mass_matrix_sqrt
            self._mass_matrix_sqrt_inverse[site_names] = mass_matrix_sqrt_inverse

    def configure(self, mass_matrix_shape, adapt_mass_matrix=True, options={}):
        """
        Setups initial mass matrix.

        :param dict mass_matrix_shape: a dict that maps tuples of site names to the shape of
            the corresponding mass matrix. Each tuple of site names corresponds to a block.
        :param bool adapt_mass_matrix: a flag to decide whether an adaptation scheme will be used.
        :param dict options: tensor options to construct the initial mass matrix.
        """
        inverse_mass_matrix = {}
        for site_names, shape in mass_matrix_shape.items():
            self._mass_matrix_size[site_names] = shape[0]
            diagonal = len(shape) == 1
            if site_names not in self._inverse_mass_matrix:
                inverse_mass_matrix[site_names] = torch.full(shape, self._init_scale, **options) \
                    if diagonal else torch.eye(*shape, **options) * self._init_scale
            else:
                # verify the shape is consistent
                assert self._inverse_mass_matrix[site_names].shape == shape
            if adapt_mass_matrix:
                adapt_scheme = WelfordCovariance(diagonal=diagonal)
                self._adapt_scheme[site_names] = adapt_scheme

        self.inverse_mass_matrix = inverse_mass_matrix

    def update(self, z, z_grad):
        """
        Updates the adaptation scheme using the new sample `z` or its grad `z_grad`.

        :param dict z: the current value.
        :param dict z_grad: grad of the current value.
        """
        for site_names, adapt_scheme in self._adapt_scheme.items():
            z_flat = torch.cat([z[name].reshape(-1) for name in site_names])
            adapt_scheme.update(z_flat.detach())

    def end_adaptation(self):
        """
        Updates the current mass matrix using the adaptation scheme.
        """
        inverse_mass_matrix = {}
        for site_names, adapt_scheme in self._adapt_scheme.items():
            inverse_mass_matrix[site_names] = adapt_scheme.get_covariance(regularize=True)
        self.inverse_mass_matrix = inverse_mass_matrix

    def kinetic_grad(self, r):
        """
        Computes the gradient of kinetic energy w.r.t. the momentum `r`.
        It is equivalent to compute velocity given the momentum `r`.

        :param dict r: a dictionary maps site names to a tensor momentum.
        :returns: a dictionary maps site names to the corresponding gradient
        """
        v = {}
        for site_names, inverse_mass_matrix in self._inverse_mass_matrix.items():
            r_flat = torch.cat([r[site_name].reshape(-1) for site_name in site_names])
            v_flat = _matmul(inverse_mass_matrix, r_flat)

            # unpacking
            pos = 0
            for site_name in site_names:
                next_pos = pos + r[site_name].numel()
                v[site_name] = v_flat[pos:next_pos].reshape(r[site_name].shape)
                pos = next_pos
        return v

    def scale(self, r_unscaled, r_prototype):
        """
        Computes `M^{1/2} @ r_unscaled`.

        Note that `r` is generated from a gaussian with scale `mass_matrix_sqrt`.
        This method will unscale it.

        :param dict r_unscaled: a dictionary maps site names to a tensor momentum.
        :param dict r_prototype: a dictionary mapes site names to prototype momentum.
            Those prototype values are used to get shapes of the scaled version.
        :returns: a dictionary maps site names to the corresponding tensor
        """
        s = {}
        for site_names, mass_matrix_sqrt in self._mass_matrix_sqrt.items():
            r_flat = _matmul(mass_matrix_sqrt, r_unscaled[site_names])

            # unpacking
            pos = 0
            for site_name in site_names:
                next_pos = pos + r_prototype[site_name].numel()
                s[site_name] = r_flat[pos:next_pos].reshape(r_prototype[site_name].shape)
                pos = next_pos
        return s

    def unscale(self, r):
        """
        Computes `inv(M^{1/2}) @ r`.

        Note that `r` is generated from a gaussian with scale `mass_matrix_sqrt`.
        This method will unscaled it.

        :param dict r: a dictionary maps site names to a tensor momentum.
        :returns: a dictionary maps site names to the corresponding tensor
        """
        u = {}
        for site_names, mass_matrix_sqrt_inverse in self._mass_matrix_sqrt_inverse.items():
            r_flat = torch.cat([r[site_name].reshape(-1) for site_name in site_names])
            u[site_names] = _matmul(mass_matrix_sqrt_inverse, r_flat)
        return u
