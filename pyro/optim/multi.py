# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch

from pyro.ops.newton import newton_step
from pyro.optim.optim import PyroOptim


class MultiOptimizer:
    """
    Base class of optimizers that make use of higher-order derivatives.

    Higher-order optimizers generally use :func:`torch.autograd.grad` rather
    than :meth:`torch.Tensor.backward`, and therefore require a different
    interface from usual Pyro and PyTorch optimizers. In this interface,
    the :meth:`step` method inputs a ``loss`` tensor to be differentiated,
    and backpropagation is triggered one or more times inside the optimizer.

    Derived classes must implement :meth:`step` to compute derivatives and
    update parameters in-place.

    Example::

        tr = poutine.trace(model).get_trace(*args, **kwargs)
        loss = -tr.log_prob_sum()
        params = {name: site['value'].unconstrained()
                  for name, site in tr.nodes.items()
                  if site['type'] == 'param'}
        optim.step(loss, params)
    """
    def step(self, loss, params):
        """
        Performs an in-place optimization step on parameters given a
        differentiable ``loss`` tensor.

        Note that this detaches the updated tensors.

        :param torch.Tensor loss: A differentiable tensor to be minimized.
            Some optimizers require this to be differentiable multiple times.
        :param dict params: A dictionary mapping param name to unconstrained
            value as stored in the param store.
        """
        updated_values = self.get_step(loss, params)
        for name, value in params.items():
            with torch.no_grad():
                # we need to detach because updated_value may depend on value
                value.copy_(updated_values[name].detach())

    def get_step(self, loss, params):
        """
        Computes an optimization step of parameters given a differentiable
        ``loss`` tensor, returning the updated values.

        Note that this preserves derivatives on the updated tensors.

        :param torch.Tensor loss: A differentiable tensor to be minimized.
            Some optimizers require this to be differentiable multiple times.
        :param dict params: A dictionary mapping param name to unconstrained
            value as stored in the param store.
        :return: A dictionary mapping param name to updated unconstrained
            value.
        :rtype: dict
        """
        raise NotImplementedError


class PyroMultiOptimizer(MultiOptimizer):
    """
    Facade to wrap :class:`~pyro.optim.optim.PyroOptim` objects
    in a :class:`MultiOptimizer` interface.
    """
    def __init__(self, optim):
        if not isinstance(optim, PyroOptim):
            raise TypeError('Expected a PyroOptim object but got a {}'.format(type(optim)))
        self.optim = optim

    def step(self, loss, params):
        values = params.values()
        grads = torch.autograd.grad(loss, values, create_graph=True)
        for x, g in zip(values, grads):
            x.grad = g
        self.optim(values)


class TorchMultiOptimizer(PyroMultiOptimizer):
    """
    Facade to wrap :class:`~torch.optim.Optimizer` objects
    in a :class:`MultiOptimizer` interface.
    """
    def __init__(self, optim_constructor, optim_args):
        optim = PyroOptim(optim_constructor, optim_args)
        super().__init__(optim)


class MixedMultiOptimizer(MultiOptimizer):
    """
    Container class to combine different :class:`MultiOptimizer` instances for
    different parameters.

    :param list parts: A list of ``(names, optim)`` pairs, where each
        ``names`` is a list of parameter names, and each ``optim`` is a
        :class:`MultiOptimizer` or :class:`~pyro.optim.optim.PyroOptim` object
        to be used for the named parameters. Together the ``names`` should
        partition up all desired parameters to optimize.
    :raises ValueError: if any name is optimized by multiple optimizers.
    """
    def __init__(self, parts):
        optim_dict = {}
        self.parts = []
        for names_part, optim in parts:
            if isinstance(optim, PyroOptim):
                optim = PyroMultiOptimizer(optim)
            for name in names_part:
                if name in optim_dict:
                    raise ValueError("Attempted to optimize parameter '{}' by two different optimizers: "
                                     "{} vs {}" .format(name, optim_dict[name], optim))
                optim_dict[name] = optim
            self.parts.append((names_part, optim))

    def step(self, loss, params):
        for names_part, optim in self.parts:
            optim.step(loss, {name: params[name] for name in names_part})

    def get_step(self, loss, params):
        updated_values = {}
        for names_part, optim in self.parts:
            updated_values.update(
                optim.get_step(loss, {name: params[name] for name in names_part}))
        return updated_values


class Newton(MultiOptimizer):
    """
    Implementation of :class:`MultiOptimizer` that performs a Newton update
    on batched low-dimensional variables, optionally regularizing via a
    per-parameter ``trust_radius``. See :func:`~pyro.ops.newton.newton_step`
    for details.

    The result of :meth:`get_step` will be differentiable, however the
    updated values from :meth:`step` will be detached.

    :param dict trust_radii: a dict mapping parameter name to radius of trust
        region. Missing names will use unregularized Newton update, equivalent
        to infinite trust radius.
    """
    def __init__(self, trust_radii={}):
        self.trust_radii = trust_radii

    def get_step(self, loss, params):
        updated_values = {}
        for name, value in params.items():
            trust_radius = self.trust_radii.get(name)
            updated_value, cov = newton_step(loss, value, trust_radius)
            updated_values[name] = updated_value
        return updated_values
