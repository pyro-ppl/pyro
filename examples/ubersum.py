"""
This example demonstrates how to use ``ubersum`` with different backends to
compute logprob, gradients, MAP estimates, posterior samples, and marginals.

The results of these computations are returned, but this script does not
make use of them; instead we simply time the operations for profiling.
All profiling is done on jit-compiled functions. We exclude jit compilation
time from profiling results, assuming this can be done once.

You can measure complexity of different ubersum problems by specifying
``--equation`` and ``--batch-dims``.
"""

from __future__ import absolute_import, division, print_function

import argparse
import timeit

import torch
from torch.autograd import grad

from pyro.ops.contract import ubersum
from pyro.ops.einsum.adjoint import require_backward
from pyro.util import ignore_jit_warnings

# We will cache jit-compiled versions of each function.
_CACHE = {}


def jit_logprob(equation, *operands, **kwargs):
    """
    Runs ubersum to compute the partition function.

    This simulates evaluating an undirected graphical model.
    """
    key = 'logprob', equation, kwargs['batch_dims']
    if key not in _CACHE:

        # This simply wraps ubersum for jit compilation.
        def _ubersum(*operands):
            return ubersum(equation, *operands, **kwargs)

        _CACHE[key] = torch.jit.trace(_ubersum, operands, check_trace=False)

    return _CACHE[key](*operands)


def jit_gradient(equation, *operands, **kwargs):
    """
    Runs ubersum and calls backward on the partition function.

    This is simulates training an undirected graphical model.
    """
    key = 'gradient', equation, kwargs['batch_dims']
    if key not in _CACHE:

        # This wraps ubersum for jit compilation, but we will call backward on the result.
        def _ubersum(*operands):
            return ubersum(equation, *operands, **kwargs)

        _CACHE[key] = torch.jit.trace(_ubersum, operands, check_trace=False)

    # Run forward pass.
    losses = _CACHE[key](*operands)

    # Work around PyTorch 1.0.0 bug https://github.com/pytorch/pytorch/issues/14875
    # whereby tuples of length 1 are unwrapped by the jit.
    if not isinstance(losses, tuple):
        losses = (losses,)

    # Run backward pass.
    grads = tuple(grad(loss, operands, retain_graph=True, allow_unused=True)
                  for loss in losses)
    return grads


def _jit_adjoint(equation, *operands, **kwargs):
    """
    Runs ubersum in forward-backward mode using ``pyro.ops.adjoint``.

    This simulates serving predictions from an undirected graphical model.
    """
    backend = kwargs.pop('backend', 'pyro.ops.einsum.torch_sample')
    key = backend, equation, tuple(x.shape for x in operands), kwargs['batch_dims']
    if key not in _CACHE:

        # This wraps a complete adjoint algorithm call.
        @ignore_jit_warnings()
        def _forward_backward(*operands):
            # First we request backward results on each input operand.
            # This is the pyro.ops.adjoint equivalent of torch's .require_grad_().
            for operand in operands:
                require_backward(operand)

            # Next we run the forward pass.
            results = ubersum(equation, *operands, backend=backend, **kwargs)

            # The we run a backward pass.
            for result in results:
                result._pyro_backward()

            # Finally we retrieve results from the ._pyro_backward_result attribute
            # that has been set on each input operand. If you only want results on a
            # subset of operands, you can call require_backward() on only those.
            results = []
            for x in operands:
                results.append(x._pyro_backward_result)
                x._pyro_backward_result = None

            return tuple(results)

        _CACHE[key] = torch.jit.trace(_forward_backward, operands, check_trace=False)

    return _CACHE[key](*operands)


def jit_map(equation, *operands, **kwargs):
    return _jit_adjoint(equation, *operands, backend='pyro.ops.einsum.torch_map', **kwargs)


def jit_sample(equation, *operands, **kwargs):
    return _jit_adjoint(equation, *operands, backend='pyro.ops.einsum.torch_sample', **kwargs)


def jit_marginal(equation, *operands, **kwargs):
    return _jit_adjoint(equation, *operands, backend='pyro.ops.einsum.torch_marginal', **kwargs)


def time_fn(fn, equation, *operands, **kwargs):
    iters = kwargs.pop('iters')
    _CACHE.clear()  # Avoid memory leaks.
    fn(equation, *operands, **kwargs)

    time_start = timeit.default_timer()
    for i in range(iters):
        fn(equation, *operands, **kwargs)
    time_end = timeit.default_timer()

    return (time_end - time_start) / iters


def main(args):
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    if args.method == 'all':
        for method in ['logprob', 'gradient', 'marginal', 'map', 'sample']:
            args.method = method
            main(args)
        return

    print('Plate size  Time per iteration of {} (ms)'.format(args.method))
    fn = globals()['jit_{}'.format(args.method)]
    equation = args.equation
    batch_dims = args.batch_dims
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')

    # Vary all plate sizes at the same time.
    for plate_size in range(8, 1 + args.max_plate_size, 8):
        operands = []
        for dims in inputs:
            shape = torch.Size([plate_size if d in batch_dims else args.dim_size
                                for d in dims])
            operands.append(torch.randn(shape, requires_grad=True))

        time = time_fn(fn, equation, *operands, batch_dims=batch_dims, modulo_total=True,
                       iters=args.iters)
        print('{: <11s} {:0.4g}'.format('{} ** {}'.format(plate_size, len(args.batch_dims)), time * 1e3))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ubersum profiler')
    parser.add_argument('-e', '--equation', default='a,abi,bcij,adj,deij->')
    parser.add_argument('-b', '--batch-dims', default='ij')
    parser.add_argument('-d', '--dim-size', default=32, type=int)
    parser.add_argument('-p', '--max-plate-size', default=32, type=int)
    parser.add_argument('-n', '--iters', default=10, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('-m', '--method', default='all',
                        help='one of: logprob, gradient, marginal, map, sample')
    args = parser.parse_args()
    main(args)
