# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
from subprocess import check_call

import pytest
import torch

from tests.common import EXAMPLES_DIR, requires_cuda, requires_horovod, xfail_param

logger = logging.getLogger(__name__)
pytestmark = pytest.mark.stage('test_examples')


CPU_EXAMPLES = [
    'air/main.py --num-steps=1',
    'air/main.py --num-steps=1 --no-baseline',
    'baseball.py --num-samples=200 --warmup-steps=100 --num-chains=2',
    'lkj.py --n=50 --num-chains=1 --warmup-steps=100 --num-samples=200',
    'capture_recapture/cjs.py --num-steps=1 -m 1',
    'capture_recapture/cjs.py --num-steps=1 -m 2',
    'capture_recapture/cjs.py --num-steps=1 -m 3',
    'capture_recapture/cjs.py --num-steps=1 -m 4',
    'capture_recapture/cjs.py --num-steps=1 -m 5',
    'capture_recapture/cjs.py --num-steps=1 -m 1 --tmc --tmc-num-samples=2',
    'capture_recapture/cjs.py --num-steps=1 -m 2 --tmc --tmc-num-samples=2',
    'capture_recapture/cjs.py --num-steps=1 -m 3 --tmc --tmc-num-samples=2',
    'capture_recapture/cjs.py --num-steps=1 -m 4 --tmc --tmc-num-samples=2',
    'capture_recapture/cjs.py --num-steps=1 -m 5 --tmc --tmc-num-samples=2',
    'contrib/autoname/scoping_mixture.py --num-epochs=1',
    'contrib/autoname/mixture.py --num-epochs=1',
    'contrib/autoname/tree_data.py --num-epochs=1',
    'contrib/cevae/synthetic.py --num-epochs=1',
    'contrib/epidemiology/sir.py --nojit -np=128 -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2',
    'contrib/epidemiology/sir.py --nojit -np=128 -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 -c=2',
    'contrib/epidemiology/sir.py --nojit -np=128 -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 -e=2',
    'contrib/epidemiology/sir.py --nojit -np=128 -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 -k=1',
    'contrib/epidemiology/sir.py --nojit -np=128 -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 -e=2 -k=1',
    'contrib/epidemiology/sir.py --nojit -np=128 -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 --haar',
    'contrib/epidemiology/sir.py --nojit -np=128 -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 -nb=4',
    'contrib/epidemiology/sir.py --nojit -np=128 -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 -hfm=3',
    'contrib/epidemiology/sir.py --nojit -np=128 -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 -a',
    'contrib/epidemiology/sir.py --nojit -np=128 -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 -o=0.2',
    'contrib/epidemiology/sir.py --nojit -np=128 -ss=2 -n=4 -d=20 -p=1000 -f 2 --svi',
    'contrib/epidemiology/regional.py --nojit -t=2 -w=2 -n=4 -r=3 -d=20 -p=1000 -f 2',
    'contrib/epidemiology/regional.py --nojit -t=2 -w=2 -n=4 -r=3 -d=20 -p=1000 -f 2 --haar',
    'contrib/epidemiology/regional.py --nojit -t=2 -w=2 -n=4 -r=3 -d=20 -p=1000 -f 2 -hfm=3',
    'contrib/epidemiology/regional.py --nojit -t=2 -w=2 -n=4 -r=3 -d=20 -p=1000 -f 2 -nb=4',
    'contrib/epidemiology/regional.py --nojit -ss=2 -n=4 -r=3 -d=20 -p=1000 -f 2 --svi',
    'contrib/forecast/bart.py --num-steps=2 --stride=99999',
    'contrib/gp/sv-dkl.py --epochs=1 --num-inducing=4 --batch-size=1000',
    'contrib/gp/sv-dkl.py --binary --epochs=1 --num-inducing=4 --batch-size=1000',
    'contrib/oed/ab_test.py --num-vi-steps=10 --num-bo-steps=2',
    'contrib/timeseries/gp_models.py -m imgp --test --num-steps=2',
    'contrib/timeseries/gp_models.py -m lcmgp --test --num-steps=2',
    'dmm/dmm.py --num-epochs=1',
    'dmm/dmm.py --num-epochs=1 --tmcelbo --tmc-num-samples=2',
    'dmm/dmm.py --num-epochs=1 --num-iafs=1',
    'dmm/dmm.py --num-epochs=1 --tmc --tmc-num-samples=2',
    'dmm/dmm.py --num-epochs=1 --tmcelbo --tmc-num-samples=2',
    'eight_schools/mcmc.py --num-samples=500 --warmup-steps=100',
    'eight_schools/svi.py --num-epochs=1',
    'einsum.py',
    'hmm.py --num-steps=1 --truncate=10 --model=0',
    'hmm.py --num-steps=1 --truncate=10 --model=1',
    'hmm.py --num-steps=1 --truncate=10 --model=2',
    'hmm.py --num-steps=1 --truncate=10 --model=3',
    'hmm.py --num-steps=1 --truncate=10 --model=4',
    'hmm.py --num-steps=1 --truncate=10 --model=5',
    'hmm.py --num-steps=1 --truncate=10 --model=6',
    'hmm.py --num-steps=1 --truncate=10 --model=6 --raftery-parameterization',
    'hmm.py --num-steps=1 --truncate=10 --model=7',
    'hmm.py --num-steps=1 --truncate=10 --model=0 --tmc --tmc-num-samples=2',
    'hmm.py --num-steps=1 --truncate=10 --model=1 --tmc --tmc-num-samples=2',
    'hmm.py --num-steps=1 --truncate=10 --model=2 --tmc --tmc-num-samples=2',
    'hmm.py --num-steps=1 --truncate=10 --model=3 --tmc --tmc-num-samples=2',
    'hmm.py --num-steps=1 --truncate=10 --model=4 --tmc --tmc-num-samples=2',
    'hmm.py --num-steps=1 --truncate=10 --model=5 --tmc --tmc-num-samples=2',
    'hmm.py --num-steps=1 --truncate=10 --model=6 --tmc --tmc-num-samples=2',
    'inclined_plane.py --num-samples=1',
    'lda.py --num-steps=2 --num-words=100 --num-docs=100 --num-words-per-doc=8',
    'minipyro.py --backend=pyro',
    'minipyro.py',
    'mixed_hmm/experiment.py --timesteps=1',
    'neutra.py -n 10 --num-warmup 10 --num-samples 10',
    'rsa/generics.py --num-samples=10',
    'rsa/hyperbole.py --price=10000',
    'rsa/schelling.py --num-samples=10',
    'rsa/schelling_false.py --num-samples=10',
    'rsa/semantic_parsing.py --num-samples=10',
    'sir_hmc.py -t=2 -w=2 -n=4 -d=2 -m=1 --enum',
    'sir_hmc.py -t=2 -w=2 -n=4 -d=2 -p=10000 --sequential',
    'sir_hmc.py -t=2 -w=2 -n=4 -d=100 -p=10000 -f 2',
    'smcfilter.py --num-timesteps=3 --num-particles=10',
    'sparse_gamma_def.py --num-epochs=2 --eval-particles=2 --eval-frequency=1 --guide custom',
    'sparse_gamma_def.py --num-epochs=2 --eval-particles=2 --eval-frequency=1 --guide auto',
    'sparse_gamma_def.py --num-epochs=2 --eval-particles=2 --eval-frequency=1 --guide easy',
    'svi_horovod.py --num-epochs=2 --size=400 --no-horovod',
    'toy_mixture_model_discrete_enumeration.py  --num-steps=1',
    xfail_param('sparse_regression.py --num-steps=2 --num-data=50 --num-dimensions 20',
                reason='https://github.com/pyro-ppl/pyro/issues/2082'),
    'vae/ss_vae_M2.py --num-epochs=1',
    'vae/ss_vae_M2.py --num-epochs=1 --aux-loss',
    'vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=parallel',
    'vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=sequential',
    'vae/vae.py --num-epochs=1',
    'vae/vae_comparison.py --num-epochs=1',
]

CUDA_EXAMPLES = [
    'air/main.py --num-steps=1 --cuda',
    'baseball.py --num-samples=200 --warmup-steps=100 --num-chains=2 --cuda',
    'contrib/cevae/synthetic.py --num-epochs=1 --cuda',
    'contrib/epidemiology/sir.py --nojit -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 --cuda',
    'contrib/epidemiology/sir.py --nojit -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 -nb=16 --cuda',
    'contrib/epidemiology/sir.py --nojit -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2 --haar --cuda',
    'contrib/epidemiology/regional.py --nojit -t=2 -w=2 -n=4 -r=3 -d=20 -p=1000 -f 2 --cuda',
    'contrib/epidemiology/regional.py --nojit -t=2 -w=2 -n=4 -r=3 -d=20 -p=1000 -f 2 --haar --cuda',
    'contrib/gp/sv-dkl.py --epochs=1 --num-inducing=4 --cuda',
    'lkj.py --n=50 --num-chains=1 --warmup-steps=100 --num-samples=200 --cuda',
    'dmm/dmm.py --num-epochs=1 --cuda',
    'dmm/dmm.py --num-epochs=1 --num-iafs=1 --cuda',
    'dmm/dmm.py --num-epochs=1 --tmc --tmc-num-samples=2 --cuda',
    'dmm/dmm.py --num-epochs=1 --tmcelbo --tmc-num-samples=2 --cuda',
    'einsum.py --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=0 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=1 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=2 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=3 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=4 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=5 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=6 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=6 --cuda --raftery-parameterization',
    'hmm.py --num-steps=1 --truncate=10 --model=7 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=0 --tmc --tmc-num-samples=2 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=1 --tmc --tmc-num-samples=2 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=2 --tmc --tmc-num-samples=2 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=3 --tmc --tmc-num-samples=2 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=4 --tmc --tmc-num-samples=2 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=5 --tmc --tmc-num-samples=2 --cuda',
    'hmm.py --num-steps=1 --truncate=10 --model=6 --tmc --tmc-num-samples=2 --cuda',
    'sir_hmc.py -t=2 -w=2 -n=4 -d=2 -m=1 --enum --cuda',
    'sir_hmc.py -t=2 -w=2 -n=4 -d=2 -p=10000 --sequential --cuda',
    'sir_hmc.py -t=2 -w=2 -n=4 -d=100 -p=10000 --cuda',
    'svi_horovod.py --num-epochs=2 --size=400 --cuda --no-horovod',
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
    'contrib/autoname/mixture.py --num-epochs=1 --jit',
    'contrib/cevae/synthetic.py --num-epochs=1 --jit',
    'contrib/epidemiology/sir.py --jit -np=128 -t=2 -w=2 -n=4 -d=20 -p=1000 -f 2',
    'contrib/epidemiology/sir.py --jit -np=128 -ss=2 -n=4 -d=20 -p=1000 -f 2 --svi',
    'contrib/epidemiology/regional.py --jit -t=2 -w=2 -n=4 -r=3 -d=20 -p=1000 -f 2',
    'contrib/epidemiology/regional.py --jit -ss=2 -n=4 -r=3 -d=20 -p=1000 -f 2 --svi',
    xfail_jit('lkj.py --n=50 --num-chains=1 --warmup-steps=100 --num-samples=200 --jit'),
    xfail_jit('contrib/gp/sv-dkl.py --epochs=1 --num-inducing=4 --jit'),
    xfail_jit('dmm/dmm.py --num-epochs=1 --jit'),
    xfail_jit('dmm/dmm.py --num-epochs=1 --num-iafs=1 --jit'),
    'eight_schools/mcmc.py --num-samples=500 --warmup-steps=100 --jit',
    'eight_schools/svi.py --num-epochs=1 --jit',
    'hmm.py --num-steps=1 --truncate=10 --model=1 --jit',
    'hmm.py --num-steps=1 --truncate=10 --model=2 --jit',
    'hmm.py --num-steps=1 --truncate=10 --model=3 --jit',
    'hmm.py --num-steps=1 --truncate=10 --model=4 --jit',
    'hmm.py --num-steps=1 --truncate=10 --model=5 --jit',
    'hmm.py --num-steps=1 --truncate=10 --model=7 --jit',
    xfail_jit('hmm.py --num-steps=1 --truncate=10 --model=1 --tmc --tmc-num-samples=2 --jit'),
    xfail_jit('hmm.py --num-steps=1 --truncate=10 --model=2 --tmc --tmc-num-samples=2 --jit'),
    xfail_jit('hmm.py --num-steps=1 --truncate=10 --model=3 --tmc --tmc-num-samples=2 --jit'),
    xfail_jit('hmm.py --num-steps=1 --truncate=10 --model=4 --tmc --tmc-num-samples=2 --jit'),
    'lda.py --num-steps=2 --num-words=100 --num-docs=100 --num-words-per-doc=8 --jit',
    'minipyro.py --backend=pyro --jit',
    'minipyro.py --jit',
    'sir_hmc.py -t=2 -w=2 -n=4 -d=2 -m=1 --enum --jit',
    'sir_hmc.py -t=2 -w=2 -n=4 -d=2 -p=10000 --sequential --jit',
    'sir_hmc.py -t=2 -w=2 -n=4 -p=10000 --jit',
    xfail_jit('vae/ss_vae_M2.py --num-epochs=1 --aux-loss --jit'),
    'vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=parallel --jit',
    'vae/ss_vae_M2.py --num-epochs=1 --enum-discrete=sequential --jit',
    'vae/ss_vae_M2.py --num-epochs=1 --jit',
    'vae/vae.py --num-epochs=1 --jit',
    'vae/vae_comparison.py --num-epochs=1 --jit',
]

HOROVOD_EXAMPLES = [
    'svi_horovod.py --num-epochs=2 --size=400',
    pytest.param('svi_horovod.py --num-epochs=2 --size=400 --cuda',
                 marks=[requires_cuda]),
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


@requires_horovod
@pytest.mark.parametrize('np', [1, 2])
@pytest.mark.parametrize('example', HOROVOD_EXAMPLES)
def test_horovod(np, example):
    if 'cuda' in example and np > torch.cuda.device_count():
        pytest.skip()
    horovodrun = 'horovodrun -np {} --mpi-args=--oversubscribe'.format(np)
    logger.info('Running:\n{} python examples/{}'.format(horovodrun, example))
    example = example.split()
    filename, args = example[0], example[1:]
    filename = os.path.join(EXAMPLES_DIR, filename)
    check_call(horovodrun.split() + [sys.executable, filename] + args)
