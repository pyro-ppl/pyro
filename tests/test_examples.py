from __future__ import absolute_import, division, print_function

import os
import sys
from subprocess import check_call

import pytest

from tests.common import EXAMPLES_DIR, requires_cuda

CPU_EXAMPLES = []
CUDA_EXAMPLES = []


def discover_examples():
    for root, dirs, files in os.walk(EXAMPLES_DIR):
        for basename in files:
            if not basename.endswith('.py'):
                continue
            path = os.path.join(root, basename)
            with open(path) as f:
                text = f.read()
            if '--num-epochs' in text:
                args = ['--num-epochs=1']
            elif '--num-steps' in text:
                args = ['--num-steps=1']
            elif '--num-samples' in text:
                args = ['--num-samples=1']
            else:
                # Either this is not a main file, or we don't know how to run it cheaply.
                continue
            example = os.path.relpath(path, EXAMPLES_DIR)
            # TODO: May be worth whitelisting the set of arguments to test
            # for each example.
            CPU_EXAMPLES.append((example, args))
            if '--aux-loss' in text:
                CPU_EXAMPLES.append((example, args + ['--aux-loss']))
            if '--enum-discrete' in text:
                CPU_EXAMPLES.append((example, args + ['--enum-discrete=sequential']))
                # TODO fix examples to work with --enum-discrete=parallel
                # CPU_EXAMPLES.append((example, args + ['--enum-discrete=parallel']))
            if '--num-iafs' in text:
                CPU_EXAMPLES.append((example, args + ['--num-iafs=1']))
            if '--cuda' in text:
                CUDA_EXAMPLES.append((example, args + ['--cuda']))
    CPU_EXAMPLES.sort()
    CUDA_EXAMPLES.sort()


discover_examples()


def make_ids(examples):
    return ['{} {}'.format(example, ' '.join(args)) for example, args in examples]


@pytest.mark.stage("test_examples")
@pytest.mark.parametrize('example,args', CPU_EXAMPLES, ids=make_ids(CPU_EXAMPLES))
def test_cpu(example, args):
    if example == 'bayesian_regression.py':
        pytest.skip("Failure on PyTorch master - https://github.com/uber/pyro/issues/953")
    example = os.path.join(EXAMPLES_DIR, example)
    check_call([sys.executable, example] + args)


@requires_cuda
@pytest.mark.stage("test_examples")
@pytest.mark.parametrize('example,args', CUDA_EXAMPLES, ids=make_ids(CUDA_EXAMPLES))
def test_cuda(example, args):
    example = os.path.join(EXAMPLES_DIR, example)
    check_call([sys.executable, example] + args)
