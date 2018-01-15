#!/usr/bin/env bash
set -xe

# 1. Install PyTorch
# Use conda package if pytorch_branch = 'release'.
# Else, install from source, using git branch `pytorch_branch`
if [ ${pytorch_branch} = "release" ]
then
    conda install -y pytorch torchvision -c pytorch
    if [ ${cuda} ]; then conda install -y cuda90 -c pytorch; fi
else
    conda install -y numpy pyyaml mkl setuptools cmake cffi
    if [ ${cuda} ]; then conda install -y cuda90 -c pytorch; fi
    git clone --recursive https://github.com/pytorch/pytorch.git
    cd pytorch && git checkout ${pytorch_branch}
    python setup.py install
    cd ..
fi


# 2. Install Pyro
# Use pypi wheel if pyro_branch = 'release'.
# Else, install from source, using git branch `pyro_branch`
if [ ${pyro_branch} = "release" ]
then
    pip install pyro-ppl
else
    conda install -y numpy pyyaml mkl setuptools cmake cffi
    conda install -y -c soumith magma-cuda90
    git clone https://github.com/uber/pyro.git
    (cd pyro && git checkout ${pyro_branch} && pip install -e .)
fi
