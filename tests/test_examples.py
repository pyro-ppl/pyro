from __future__ import absolute_import, division, print_function

import os
import sys
from subprocess import check_call

import pytest

from tests.common import EXAMPLES_DIR, requires_cuda

pytestmark = pytest.mark.stage('test_examples')


CPU_EXAMPLES = [
    ['air/main.py', '--num-steps=1'],
    ['baseball.py', '--num-samples=200', '--warmup-steps=100'],
    ['bayesian_regression.py', '--num-epochs=1'],
    ['contrib/autoname/scoping_mixture.py', '--num-epochs=1'],
    ['contrib/autoname/mixture.py', '--num-epochs=1'],
    ['contrib/autoname/tree_data.py', '--num-epochs=1'],
    ['contrib/gp/sv-dkl.py', '--epochs=1', '--num-inducing=4'],
    ['dmm/dmm.py', '--num-epochs=1'],
    ['dmm/dmm.py', '--num-epochs=1', '--num-iafs=1'],
    ['eight_schools/mcmc.py', '--num-samples=500', '--warmup-steps=100'],
    ['eight_schools/svi.py', '--num-epochs=1'],
    ['inclined_plane.py', '--num-samples=1'],
    ['rsa/generics.py', '--num-samples=10'],
    ['rsa/hyperbole.py', '--price=10000'],
    ['rsa/schelling.py', '--num-samples=10'],
    ['rsa/schelling_false.py', '--num-samples=10'],
    ['rsa/semantic_parsing.py', '--num-samples=10'],
    ['sparse_gamma_def.py', '--num-epochs=1'],
    ['vae/ss_vae_M2.py', '--num-epochs=1'],
    ['vae/ss_vae_M2.py', '--num-epochs=1', '--aux-loss'],
    ['vae/ss_vae_M2.py', '--num-epochs=1', '--enum-discrete=parallel'],
    ['vae/ss_vae_M2.py', '--num-epochs=1', '--enum-discrete=sequential'],
    ['vae/vae.py', '--num-epochs=1'],
    ['vae/vae_comparison.py', '--num-epochs=1'],
]

CUDA_EXAMPLES = [
    ['air/main.py', '--num-steps=1', '--cuda'],
    ['bayesian_regression.py', '--num-epochs=1', '--cuda'],
    ['contrib/gp/sv-dkl.py', '--epochs=1', '--num-inducing=4', '--cuda'],
    ['dmm/dmm.py', '--num-epochs=1', '--cuda'],
    ['dmm/dmm.py', '--num-epochs=1', '--num-iafs=1', '--cuda'],
    ['vae/vae.py', '--num-epochs=1', '--cuda'],
    ['vae/ss_vae_M2.py', '--num-epochs=1', '--cuda'],
    ['vae/ss_vae_M2.py', '--num-epochs=1', '--aux-loss', '--cuda'],
    ['vae/ss_vae_M2.py', '--num-epochs=1', '--enum-discrete=parallel', '--cuda'],
    ['vae/ss_vae_M2.py', '--num-epochs=1', '--enum-discrete=sequential', '--cuda'],
]

CPU_EXAMPLES = [(example[0], example[1:]) for example in sorted(CPU_EXAMPLES)]
CUDA_EXAMPLES = [(example[0], example[1:]) for example in sorted(CUDA_EXAMPLES)]


def make_ids(examples):
    return ['{} {}'.format(example, ' '.join(args)) for example, args in examples]


def test_coverage():
    cpu_tests = set([name for name, _ in CPU_EXAMPLES])
    cuda_tests = set([name for name, _ in CUDA_EXAMPLES])
    for root, dirs, files in os.walk(EXAMPLES_DIR):
        for basename in files:
            if not basename.endswith('.py'):
                continue
            path = os.path.join(root, basename)
            with open(path) as f:
                text = f.read()
            example = os.path.relpath(path, EXAMPLES_DIR)
            if '__main__' in text:
                if example not in cpu_tests:
                    pytest.fail('Example: {} not covered in CPU_TESTS.'.format(example))
                if '--cuda' in text and example not in cuda_tests:
                    pytest.fail('Example: {} not covered by CUDA_TESTS.'.format(example))


@pytest.mark.parametrize('example,args', CPU_EXAMPLES, ids=make_ids(CPU_EXAMPLES))
def test_cpu(example, args):
    example = os.path.join(EXAMPLES_DIR, example)
    check_call([sys.executable, example] + args)


@requires_cuda
@pytest.mark.parametrize('example,args', CUDA_EXAMPLES, ids=make_ids(CUDA_EXAMPLES))
def test_cuda(example, args):
    example = os.path.join(EXAMPLES_DIR, example)
    check_call([sys.executable, example] + args)
