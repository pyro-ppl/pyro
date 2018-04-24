from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.autograd import Variable

from profiler.profiling_utils import Profile, profile_print
from pyro.distributions import (Bernoulli, Beta, Categorical, Cauchy, Dirichlet, Exponential, Gamma, LogNormal, Normal,
                                OneHotCategorical, Poisson, Uniform)


def T(arr):
    return Variable(torch.DoubleTensor(arr))


TOOL = 'timeit'
TOOL_CFG = {}
DISTRIBUTIONS = {
    'Bernoulli': (Bernoulli, {
        'probs': T([0.3, 0.3, 0.3, 0.3])
    }),
    'Beta': (Beta, {
        'concentration1': T([2.4, 2.4, 2.4, 2.4]),
        'concentration0': T([3.2, 3.2, 3.2, 3.2])
    }),
    'Categorical': (Categorical, {
        'probs': T([0.1, 0.3, 0.4, 0.2])
    }),
    'OneHotCategorical': (OneHotCategorical, {
        'probs': T([0.1, 0.3, 0.4, 0.2])
    }),
    'Dirichlet': (Dirichlet, {
        'concentration': T([2.4, 3, 6, 6])
    }),
    'Normal': (Normal, {
        'loc': T([0.5, 0.5, 0.5, 0.5]),
        'scale': T([1.2, 1.2, 1.2, 1.2])
    }),
    'LogNormal': (LogNormal, {
        'loc': T([0.5, 0.5, 0.5, 0.5]),
        'scale': T([1.2, 1.2, 1.2, 1.2])
    }),
    'Cauchy': (Cauchy, {
        'loc': T([0.5, 0.5, 0.5, 0.5]),
        'scale': T([1.2, 1.2, 1.2, 1.2])
    }),
    'Exponential': (Exponential, {
        'rate': T([5.5, 3.2, 4.1, 5.6])
    }),
    'Poisson': (Poisson, {
        'rate': T([5.5, 3.2, 4.1, 5.6])
    }),
    'Gamma': (Gamma, {
        'concentration': T([2.4, 2.4, 2.4, 2.4]),
        'rate': T([3.2, 3.2, 3.2, 3.2])
    }),
    'Uniform': (Uniform, {
        'low': T([0, 0, 0, 0]),
        'high': T([4, 4, 4, 4])
    })
}


def get_tool():
    return TOOL


def get_tool_cfg():
    return TOOL_CFG


@Profile(
    tool=get_tool,
    tool_cfg=get_tool_cfg,
    fn_id=lambda dist, batch_size, *args, **kwargs: 'sample_' + dist.dist_class.__name__ + '_N=' + str(batch_size))
def sample(dist, batch_size):
    return dist.sample(sample_shape=(batch_size,))


@Profile(
    tool=get_tool,
    tool_cfg=get_tool_cfg,
    fn_id=lambda dist, batch, *args, **kwargs:  #
    'log_prob_' + dist.dist_class.__name__ + '_N=' + str(batch.size()[0]))
def log_prob(dist, batch):
    return dist.log_prob(batch)


def run_with_tool(tool, dists, batch_sizes):
    column_widths, field_format, template = None, None, None
    if tool == 'timeit':
        profile_cols = 2 * len(batch_sizes)
        column_widths = [14] * (profile_cols + 1)
        field_format = [None] + ['{:.6f}'] * profile_cols
        template = 'column'
    elif tool == 'cprofile':
        column_widths = [14, 80]
        template = 'row'
    with profile_print(column_widths, field_format, template) as out:
        column_headers = []
        for size in batch_sizes:
            column_headers += ['SAMPLE (N=' + str(size) + ')', 'LOG_PROB (N=' + str(size) + ')']
        out.header(['DISTRIBUTION'] + column_headers)
        for dist_name in dists:
            Dist, params = DISTRIBUTIONS[dist_name]
            result_row = [dist_name]
            dist = Dist(**params)
            for size in batch_sizes:
                sample_result, sample_prof = sample(dist, batch_size=size)
                _, logpdf_prof = log_prob(dist, sample_result)
                result_row += [sample_prof, logpdf_prof]
            out.push(result_row)


def set_tool_cfg(args):
    global TOOL, TOOL_CFG
    TOOL = args.tool
    tool_cfg = {}
    if args.tool == 'timeit':
        repeat = 5
        if args.repeat is not None:
            repeat = args.repeat
        tool_cfg = {'repeat': repeat}
    TOOL_CFG = tool_cfg


def main():
    parser = argparse.ArgumentParser(description='Profiling distributions library using various' 'tools.')
    parser.add_argument(
        '--tool',
        nargs='?',
        default='timeit',
        help='Profile using tool. One of following should be specified:'
        ' ["timeit", "cprofile"]')
    parser.add_argument(
        '--batch_sizes',
        nargs='*',
        type=int,
        help='Batch size of tensor - max of 4 values allowed. '
        'Default = [10000, 100000]')
    parser.add_argument(
        '--dists',
        nargs='*',
        type=str,
        help='Run tests on distributions. One or more of following distributions '
        'are supported: ["bernoulli, "beta", "categorical", "dirichlet", '
        '"normal", "lognormal", "halfcauchy", "cauchy", "exponential", '
        '"poisson", "one_hot_categorical", "gamma", "uniform"] '
        'Default - Run profiling on all distributions')
    parser.add_argument(
        '--repeat',
        nargs='?',
        default=5,
        type=int,
        help='When profiling using "timeit", the number of repetitions to '
        'use for the profiled function. default=5. The minimum value '
        'is reported.')
    args = parser.parse_args()
    set_tool_cfg(args)
    dists = args.dists
    batch_sizes = args.batch_sizes
    if not args.batch_sizes:
        batch_sizes = [10000, 100000]
    if len(batch_sizes) >= 4:
        raise ValueError("Max of 4 batch sizes can be specified.")
    if not dists:
        dists = sorted(DISTRIBUTIONS.keys())
    run_with_tool(args.tool, dists, batch_sizes)


if __name__ == '__main__':
    main()
