from __future__ import absolute_import, division, print_function

import os
import sys
from collections import OrderedDict
from subprocess import check_call

import pytest

from tests.common import EXAMPLES_DIR, requires_cuda

CPU_EXAMPLES = OrderedDict()
CUDA_EXAMPLES = OrderedDict()


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
            else:
                # Either this is not a main file, or we don't know how to run it cheaply.
                continue
            example = os.path.relpath(path, EXAMPLES_DIR)
            CPU_EXAMPLES[example] = args
            if '--cuda' in text:
                CUDA_EXAMPLES[example] = [args] + ['--cuda']


discover_examples()


@pytest.mark.stage("test_examples")
@pytest.mark.parametrize('example,args', CPU_EXAMPLES.items(), ids=list(CPU_EXAMPLES))
def test_cpu(example, args):
    example = os.path.join(EXAMPLES_DIR, example)
    check_call([sys.executable, example] + args)


@requires_cuda
@pytest.mark.stage("test_examples")
@pytest.mark.parametrize('example,args', CUDA_EXAMPLES.items(), ids=list(CUDA_EXAMPLES))
def test_cuda(example, args):
    example = os.path.join(EXAMPLES_DIR, example)
    check_call([sys.executable, example] + args)
