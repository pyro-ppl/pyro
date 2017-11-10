from __future__ import absolute_import, division, print_function

import argparse

import torch
from torch.autograd import Variable

from profiler.profiling_utils import Profile, profile_print
from pyro.distributions import (bernoulli, beta, categorical, cauchy, delta, dirichlet, exponential, gamma, halfcauchy,
                                lognormal, normal, poisson, uniform)


def T(arr):
    return Variable(torch.Tensor(arr))


TOOL = 'timeit'
TOOL_CFG = {}
DISTRIBUTIONS = {
    'bernoulli': (bernoulli, {
        'ps': T([0.3, 0.3, 0.3, 0.3])
    }),
    'beta': (beta, {
        'alpha': T([2.4, 2.4, 2.4, 2.4]),
        'beta': T([3.2, 3.2, 3.2, 3.2])
    }),
    'categorical': (categorical, {
        'ps': T([0.1, 0.3, 0.4, 0.2])
    }),
    'dirichlet': (dirichlet, {
        'alpha': T([2.4, 3, 6, 6])
    }),
    'normal': (normal, {
        'mu': T([0.5, 0.5, 0.5, 0.5]),
        'sigma': T([1.2, 1.2, 1.2, 1.2])
    }),
    'lognormal': (lognormal, {
        'mu': T([0.5, 0.5, 0.5, 0.5]),
        'sigma': T([1.2, 1.2, 1.2, 1.2])
    }),
    'halfcauchy': (halfcauchy, {
        'mu': T([0.5, 0.5, 0.5, 0.5]),
        'gamma': T([1.2, 1.2, 1.2, 1.2])
    }),
    'cauchy': (cauchy, {
        'mu': T([0.5, 0.5, 0.5, 0.5]),
        'gamma': T([1.2, 1.2, 1.2, 1.2])
    }),
    'exponential': (exponential, {
        'lam': T([5.5, 3.2, 4.1, 5.6])
    }),
    'poisson': (poisson, {
        'lam': T([5.5, 3.2, 4.1, 5.6])
    }),
    'gamma': (gamma, {
        'alpha': T([2.4, 2.4, 2.4, 2.4]),
        'beta': T([3.2, 3.2, 3.2, 3.2])
    }),
    'uniform': (uniform, {
        'a': T([0, 0, 0, 0]),
        'b': T([4, 4, 4, 4])
    })
}


def get_tool():
    return TOOL


def get_tool_cfg():
    return TOOL_CFG


@Profile(
    tool=get_tool,
    tool_cfg=get_tool_cfg,
    fn_id=lambda dist, batch_size, *args, **kwargs: 'sample_' + dist.dist_class.__name__)
def sample(dist, batch_size, *args, **kwargs):
    return dist.sample(batch_size=batch_size, *args, **kwargs)


@Profile(
    tool=get_tool,
    tool_cfg=get_tool_cfg,
    fn_id=lambda dist, batch_size, *args, **kwargs: 'batch_log_pdf_' + dist.dist_class.__name__)
def batch_log_pdf(dist, batch, *args, **kwargs):
    return dist.batch_log_pdf(batch, *args, **kwargs)


def run_with_tool(tool, dists, batch_size):
    column_widths, field_format, template = None, None, None
    if tool == 'timeit':
        field_format = (None, '{:.6f}', '{:.6f}')
        template = 'column'
    elif tool == 'cprofile':
        column_widths = [16, 80]
        template = 'row'
    with profile_print(column_widths, field_format, template) as out:
        out.header('DISTRIBUTION', 'SAMPLE', 'BATCH_LOG_PDF')
        for dist_name in dists:
            dist, params = DISTRIBUTIONS[dist_name]
            sample_prof, sample_result = sample(dist, batch_size=batch_size, **params)
            logpdf_prof, _ = batch_log_pdf(dist, sample_result, **params)
            out.push(dist_name, sample_prof, logpdf_prof)


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
    parser = argparse.ArgumentParser(description='Profiling distributions library')
    parser.add_argument('--tool', nargs='?', default='timeit')
    parser.add_argument('--batch_size', nargs='?', default=100000, type=int)
    parser.add_argument('--dists', nargs='*')
    parser.add_argument('--repeat', nargs='?', default=5, type=int)
    args = parser.parse_args()
    set_tool_cfg(args)
    dists = args.dists
    if not dists:
        dists = sorted(DISTRIBUTIONS.keys())
    run_with_tool(args.tool, dists, args.batch_size)


if __name__ == '__main__':
    main()
