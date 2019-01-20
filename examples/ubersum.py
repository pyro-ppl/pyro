from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import timeit

from scipy.stats import trim_mean
import torch

from pyro.ops.contract import ubersum
from pyro.ops.einsum.adjoint import require_backward
from pyro.util import ignore_jit_warnings

_CACHE = {}


def jit_ubersum(equation, *operands, **kwargs):
    """
    Runs ubersum to compute the partition function, to simulate evaluating a model.
    """
    key = 'ubersum', equation, kwargs['batch_dims']
    if key not in _CACHE:

        def _ubersum(*operands):
            return ubersum(equation, *operands, **kwargs)

        fn = torch.jit.trace(_ubersum, operands, check_trace=False)
        _CACHE[key] = fn

    return _CACHE[key](*operands)


def jit_train(equation, *operands, **kwargs):
    """
    Runs ubersum and calls backward on the partition function, to simulate training a model.
    """
    key = 'train', equation, kwargs['batch_dims']
    if key not in _CACHE:

        def _ubersum(*operands):
            return ubersum(equation, *operands, **kwargs)

        fn = torch.jit.trace(_ubersum, operands, check_trace=False)
        _CACHE[key] = fn

    # Run forward pass.
    loss = _CACHE[key](*operands)

    # Run backward pass.
    loss.backward()  # Note loss must be differentiable.


def jit_serve(equation, *operands, **kwargs):
    """
    Runs ubersum in forward-filter backward-sample mode, to simulate serving a model.
    """
    backend = kwargs.pop('backend', 'pyro.ops.einsum.torch_sample')
    key = backend, equation, tuple(x.shape for x in operands), kwargs['batch_dims']
    if key not in _CACHE:

        @ignore_jit_warnings()
        def _sample(*operands):
            for operand in operands:
                require_backward(operand)

            # Run forward pass.
            results = ubersum(equation, *operands, backend=backend, **kwargs)

            # Run backward pass.
            for result in results:
                result._pyro_backward()

            # Retrieve results.
            results = []
            for x in operands:
                results.append(x._pyro_backward_result)
                x._pyro_backward_result = None
            return tuple(results)

        fn = torch.jit.trace(_sample, operands, check_trace=False)
        _CACHE[key] = fn

    return _CACHE[key](*operands)


def jit_map(equation, *operands, **kwargs):
    jit_serve(equation, *operands, backend='pyro.ops.einsum.torch_map', **kwargs)


def jit_sample(equation, *operands, **kwargs):
    jit_serve(equation, *operands, backend='pyro.ops.einsum.torch_sample', **kwargs)


def jit_marginal(equation, *operands, **kwargs):
    jit_serve(equation, *operands, backend='pyro.ops.einsum.torch_marginal', **kwargs)


def time_fn(fn, equation, *operands, **kwargs):
    iters = kwargs.pop('iters')
    fn(equation, *operands, **kwargs)
    times = []
    for i in range(iters):
        time_start = timeit.default_timer()
        fn(equation, *operands, **kwargs)
        time_end = timeit.default_timer()
        times.append(time_end - time_start)
    return trim_mean(times, 0.1)


def main(args):
    fn = globals()['jit_{}'.format(args.method)]
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    equation = args.equation
    batch_dims = args.batch_dims
    inputs, outputs = equation.split('->')
    inputs = inputs.split(',')

    dim_size = args.dim_size
    times = {}
    file = os.path.join(args.outdir, args.method + ".csv")
    with open(file, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["plate_size", "time"])
        for plate_size in range(8, 1 + args.max_plate_size, 8):
            _CACHE.clear()
            operands = []
            for dims in inputs:
                shape = torch.Size([plate_size if d in batch_dims else dim_size
                                    for d in dims])
                operands.append(torch.randn(shape, requires_grad=True))

            time = time_fn(fn, equation, *operands, batch_dims=batch_dims, iters=args.iters)
            times[plate_size] = time
            print('{}\t{}'.format(plate_size, time))
            writer.writerow([plate_size, time])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ubersum profiler")
    parser.add_argument("-e", "--equation", default="a,abi,bcij,adj,deij->")
    parser.add_argument("-b", "--batch-dims", default="ij")
    parser.add_argument("-d", "--dim-size", default=32, type=int)
    parser.add_argument("-p", "--max-plate-size", default=32, type=int)
    parser.add_argument("-n", "--iters", default=10000, type=int)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true', default=True)
    parser.add_argument('--outdir', type=str)
    parser.add_argument("-m", "--method", default="ubersum",
                        help="one of: ubersum, train, marginal, map, sample")
    args = parser.parse_args()
    main(args)
