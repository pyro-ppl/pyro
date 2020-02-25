# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from torch.nn.functional import pad
from torch.optim.optimizer import Optimizer

from pyro.ops.tensor_utils import dct, idct, next_fast_len


def _transform_forward(x, dim, duration):
    assert not x.requires_grad
    assert dim < 0
    assert duration == x.size(dim)
    time_domain = x
    new_size = next_fast_len(duration)
    if new_size == duration:
        freq_domain = x
    else:
        freq_domain = pad(x, (0, 0) * (-1 - dim) + (0, new_size - duration))
    freq_domain = dct(freq_domain, dim)
    return torch.cat([time_domain, freq_domain], dim=dim)


def _transform_inverse(x, dim, duration):
    assert not x.requires_grad
    assert dim < 0
    dots = (slice(None),) * (x.dim() + dim)
    left = dots + (slice(None, duration),)
    right = dots + (slice(duration, None),)
    return idct(x[right], dim)[left].add_(x[left])


class DCTAdam(Optimizer):
    """
    EXPERIMENTAL Discrete Cosine Transform-augmented
    :class:`~pyro.optim.clipped_adam.ClippedAdam` optimizer.

    This acts like :class:`~pyro.optim.clipped_adam.ClippedAdam` on most
    parameters, but if a parameter has an attribute ``._pyro_dct_dim``
    indicating a time dimension, this creates a secondary optimize in the
    frequency domain. This is useful for parameters of time series models.

    :param params: iterable of parameters to optimize or dicts defining parameter groups
    :param float lr: learning rate (default: 1e-3)
    :param tuple betas: coefficients used for computing
        running averages of gradient and its square (default: (0.9, 0.999))
    :param float eps: term added to the denominator to improve
        numerical stability (default: 1e-8)
    :param float clip_norm: magnitude of norm to which gradients are clipped (default: 10.0)
    :param float lrd: rate at which learning rate decays (default: 1.0)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 clip_norm=10.0, lrd=1.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, clip_norm=clip_norm, lrd=lrd)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        :param closure: An optional closure that reevaluates the model and returns the loss.

        Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            group['lr'] *= group['lrd']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                grad.clamp_(-group['clip_norm'], group['clip_norm'])

                # Transform selected parameters via dct.
                time_dim = getattr(p, "_pyro_dct_dim", None)
                if time_dim is not None:
                    duration = p.size(time_dim)
                    grad = _transform_forward(grad, time_dim, duration)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                if time_dim is None:
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step = _transform_inverse(exp_avg / denom, time_dim, duration)
                    p.data.add_(step.mul_(-step_size))

        return loss
