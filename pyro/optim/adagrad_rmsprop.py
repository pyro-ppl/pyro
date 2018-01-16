from __future__ import absolute_import, division, print_function

import torch
from torch.optim.optimizer import Optimizer


class AdagradRMSProp(Optimizer):
    """
    Implements a mash-up of the Adagrad algorithm and RMSProp. See ref. [1].

    References:
    [1] 'Automatic Differentiation Variational Inference', Alp Kucukelbir,
        Dustin Tran, Rajesh Ranganath, Andrew Gelman, David M. Blei
    [2] 'Lecture 6.5 RmsProp: Divide the gradient by a running average
        of its recent magnitude', Tieleman, T. and Hinton, G.,
        COURSERA: Neural Networks for Machine Learning.
    [3] 'Adaptive subgradient methods for online learning and stochastic optimization',
        Duchi, John, Hazan, E and Singer, Y.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        eta (float, optional): sets the step size scale (default: 1.0)
        t (float, optional): momentum parameter (default: 0.1)
        delta (float, optional): modulates the exponent that controls how the
            step size decales (default: 1e-16)
    """

    def __init__(self, params, eta=1.0, delta=1.0e-16, t=0.1):
        defaults = dict(eta=eta, delta=delta, t=t)
        super(AdagradRMSProp, self).__init__(params, defaults)

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
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
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
                state['sum'] *= (1.0 - group['t'])
                state['sum'] += group['t'] * grad * grad

                lr = group['eta'] * (state['step'] ** (-0.5 + group['delta']))
                std = state['sum'].sqrt()
                p.data.addcdiv_(-lr, grad, 1.0 + std)

        return loss
