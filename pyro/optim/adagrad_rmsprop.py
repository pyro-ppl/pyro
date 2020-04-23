# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.optim.optimizer import Optimizer


class AdagradRMSProp(Optimizer):
    """
    Implements a mash-up of the Adagrad algorithm and RMSProp. For the precise
    update equation see equations 10 and 11 in reference [1].

    References:
    [1] 'Automatic Differentiation Variational Inference', Alp Kucukelbir,
    Dustin Tran, Rajesh Ranganath, Andrew Gelman, David M. Blei
    URL: https://arxiv.org/abs/1603.00788
    [2] 'Lecture 6.5 RmsProp: Divide the gradient by a running average
    of its recent magnitude', Tieleman, T. and Hinton, G.,
    COURSERA: Neural Networks for Machine Learning.
    [3] 'Adaptive subgradient methods for online learning and stochastic optimization',
    Duchi, John, Hazan, E and Singer, Y.

    Arguments:

    :param params: iterable of parameters to optimize or dicts defining parameter groups
    :param eta: sets the step size scale (optional; default: 1.0)
    :type eta: float
    :param t:  t, optional): momentum parameter (optional; default: 0.1)
    :type t: float
    :param delta: modulates the exponent that controls how the step size scales (optional: default: 1e-16)
    :type delta: float
    """

    def __init__(self, params, eta=1.0, delta=1.0e-16, t=0.1):
        defaults = dict(eta=eta, delta=delta, t=t)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.zeros_like(p.data)

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """
        Performs a single optimization step.

        :param closure: A (optional) closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data

                if grad.is_sparse:
                    raise NotImplementedError

                state = self.state[p]
                state['step'] += 1
                if state['step'] == 1:
                    # if first step, initialize variance bit to grad^2
                    state['sum'] = grad * grad
                else:
                    state['sum'] *= (1.0 - group['t'])
                    state['sum'] += group['t'] * grad * grad

                lr = group['eta'] * (state['step'] ** (-0.5 + group['delta']))
                std = state['sum'].sqrt()
                p.data.addcdiv_(grad, 1.0 + std, value=-lr)

        return loss
