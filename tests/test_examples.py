import os
import sys
from subprocess import check_call

from tests.common import EXAMPLES_DIR, requires_cuda
import pytest


def find_examples_matching(*substrings):
    for root, dirs, files in os.walk(EXAMPLES_DIR):
        for basename in files:
            if not basename.endswith('.py'):
                continue
            path = os.path.join(root, basename)
            with open(path) as f:
                text = f.read()
            if all(s in text for s in substrings):
                yield os.path.relpath(path, EXAMPLES_DIR)


@pytest.mark.stage("test_examples")
@pytest.mark.parametrize('example', find_examples_matching('--num-epochs'))
def test_cpu(example):
    example = os.path.join(EXAMPLES_DIR, example)
    check_call([sys.executable, example, '--num-epochs', '1'])


@requires_cuda
@pytest.mark.stage("test_examples")
@pytest.mark.parametrize('example', find_examples_matching('--num-epochs', '--cuda'))
def test_cuda(example):
    example = os.path.join(EXAMPLES_DIR, example)
    check_call([sys.executable, example, '--num-epochs', '1', '--cuda'])
