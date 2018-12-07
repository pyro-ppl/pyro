from __future__ import absolute_import, division, print_function

import logging
import os
import sys
from subprocess import check_call

import pytest

from tests.common import EXAMPLES_DIR, requires_cuda, xfail_param, skipif_param

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.stage('test_examples')


CPU_EXAMPLES = [
    'air/main.py --num-steps=1',
    'air/main.py --num-steps=1 --no-baseline',
    'baseball.py --num-samples=200 --warmup-steps=100 --num-chains=2',
    'bayesian_regression.py --num-epochs=1',
    'contrib/autoname/scoping_mixture.py --num-epochs=1',
    'contrib/autoname/mixture.py --num-epochs=1',
    'contrib/autoname/tree_data.py --num-epochs=1',
    skipif_param('contrib/gp/sv-dkl.py --epochs=1 --num-inducing=4',
                 condition='CI' in os.environ,
                 reason='https://github.com/uber/pyro/issues/1540'),
    'contrib/oed/ab_test.py --num-vi-steps=10 --num-bo-steps=2',
    'contrib/oed/item_response.py -N=1000 -M=1000',
    'contrib/oed/sequential_oed_sigmoid_lm.py --num-experiments=2 --num-runs=2 --no-plot',
    'dmm/dmm.py --num-epochs=1',
    'dmm/dmm.py --num-epochs=1 --num-iafs=1',
    'eight_schools/mcmc.py --num-samples=500 --warmup-steps=100',
    'eight_schools/svi.py --num-epochs=1',
    'hmm.py --num-steps=1 --truncate=10 --model=0',
    'hmm.py --num-steps=1 --truncate=10 --model=1',
    'hmm.py --num-steps=1 --truncate=10 --model=2',
    'hmm.py --num-steps=1 --truncate=10 --model=3',
    'hmm.py --num-steps=1 --truncate=10 --model=4',
    'hmm.py --num-steps=1 --truncate=10 --model=5',
    'inclined_plane.py --num-samples=1',
    'lda.py --num-steps=2 --num-words=100 --num-docs=100 --num-words-per-doc=8',
    'minipyro.py',
    'minipyro.py --full-pyro',
    'rsa/generics.py --num-samples=10',
    'rsa/hyperbole.py --price=10000',
    'rsa/schelling.py --num-samples=10',
    'rsa/schelling_false.py --num-samples=10',
    'rsa/semantic_parsing.py --num-samples=10',
    'sparse_gamma_def.py --num-epochs=2 --eval-particles=2 --eval-frequency=1',
    'sparse_gamma_def.py --num-epochs=2 --eval-particles=2 --eval-frequency=1 --auto-guide',
    'vae/ss_vae_M2.py --num-epochs=1',
    'vae/ss_vae_M2.py --num-epochs=1 --aux-loss',
    'vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=parallel',
    'vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=sequential',
    'vae/vae.py --num-epochs=1',
    'vae/vae_comparison.py --num-epochs=1',
]

CUDA_EXAMPLES = [
    'air/main.py --num-steps=1 --cuda',
    xfail_param('baseball.py --num-samples=200 --warmup-steps=100 --num-chains=2 --cuda',
                reason="https://github.com/pytorch/pytorch/issues/10375"),
    'bayesian_regression.py --num-epochs=1 --cuda',
    'contrib/gp/sv-dkl.py --epochs=1 --num-inducing=4 --cuda',
    'dmm/dmm.py --num-epochs=1 --cuda',
    'dmm/dmm.py --num-epochs=1 --num-iafs=1 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=0 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=1 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=2 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=3 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=4 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=5 --cuda',
    'vae/vae.py --num-epochs=1 --cuda',
    'vae/ss_vae_M2.py --num-epochs=1 --cuda',
    'vae/ss_vae_M2.py --num-epochs=1 --aux-loss --cuda',
    'vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=parallel --cuda',
    'vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=sequential --cuda',
]


def xfail_jit(*args):
    return pytest.param(*args, marks=[pytest.mark.xfail(reason="not jittable"),
                                      pytest.mark.skipif('CI' in os.environ, reason='slow test')])


JIT_EXAMPLES = [
    'air/main.py --num-steps=1 --jit',
    'baseball.py --num-samples=200 --warmup-steps=100 --jit',
    'bayesian_regression.py --num-epochs=1 --jit',
    'contrib/autoname/mixture.py --num-epochs=1 --jit',
    xfail_jit('contrib/gp/sv-dkl.py --epochs=1 --num-inducing=4 --jit'),
    xfail_jit('dmm/dmm.py --num-epochs=1 --jit'),
    xfail_jit('dmm/dmm.py --num-epochs=1 --num-iafs=1 --jit'),
    skipif_param('eight_schools/mcmc.py --num-samples=500 --warmup-steps=100 --jit',
                 condition=True, reason='very slow test'),
    'eight_schools/svi.py --num-epochs=1 --jit',
    'hmm.py --num-steps=1 --truncate=10 --model=1 --jit',
    'hmm.py --num-steps=1 --truncate=10 --model=2 --jit',
    'hmm.py --num-steps=1 --truncate=10 --model=3 --jit',
    'hmm.py --num-steps=1 --truncate=10 --model=4 --jit',
    'hmm.py --num-steps=1 --truncate=10 --model=5 --jit',
    xfail_jit('lda.py --num-steps=2 --num-words=100 --num-docs=100 --num-words-per-doc=8 --jit'),
    xfail_jit('vae/ss_vae_M2.py --num-epochs=1 --aux-loss --jit'),
    'vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=parallel --jit',
    'vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=sequential --jit',
    'vae/ss_vae_M2.py --num-epochs=1 --jit',
    'vae/vae.py --num-epochs=1 --jit',
    'vae/vae_comparison.py --num-epochs=1 --jit',
]


def test_coverage():
    cpu_tests = set((e if isinstance(e, str) else e.values[0]).split()[0] for e in CPU_EXAMPLES)
    cuda_tests = set((e if isinstance(e, str) else e.values[0]).split()[0] for e in CUDA_EXAMPLES)
    jit_tests = set((e if isinstance(e, str) else e.values[0]).split()[0] for e in JIT_EXAMPLES)
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
                    pytest.fail('Example: {} not covered in CPU_EXAMPLES.'.format(example))
                if '--cuda' in text and example not in cuda_tests:
                    pytest.fail('Example: {} not covered by CUDA_EXAMPLES.'.format(example))
                if '--jit' in text and example not in jit_tests:
                    pytest.fail('Example: {} not covered by JIT_EXAMPLES.'.format(example))


@pytest.mark.parametrize('example', CPU_EXAMPLES)
def test_cpu(example):
    logger.info('Running:\npython examples/{}'.format(example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)


@requires_cuda
@pytest.mark.parametrize('example', CUDA_EXAMPLES)
def test_cuda(example):
    logger.info('Running:\npython examples/{}'.format(example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)


@pytest.mark.parametrize('example', JIT_EXAMPLES)
def test_jit(example):
    logger.info('Running:\npython examples/{}'.format(example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call([sys.executable, filename] + args)
