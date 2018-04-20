#!/usr/bin/env bash
set -xe

pip install --upgrade pip
pip install jupyter matplotlib

# 1. Install PyTorch
# Use conda package if pytorch_branch = 'release'.
# Else, install from source, using git branch `pytorch_branch`

if [ ${pytorch_branch} = "release" ]
then
    conda install -y pytorch torchvision -c pytorch
    if [ ${cuda} = 1 ]; then conda install -y cuda90 -c pytorch; fi
else
    conda install -y numpy pyyaml mkl mkl-include setuptools cmake cffi typing
    if [ ${cuda} = 1 ]; then conda install -y cuda90 -c pytorch; fi
    git clone --recursive https://github.com/pytorch/pytorch.git
    pushd pytorch && git checkout ${pytorch_branch}
    python setup.py install
    popd
fi


# 2. Install Pyro
# Use pypi wheel if pyro_branch = 'release'.
# Else, install from source, using git branch `pyro_branch`
if [ ${pyro_branch} = "release" ]
then
    pip install pyro-ppl
else
    git clone https://github.com/uber/pyro.git
    (cd pyro && git checkout ${pyro_branch} && pip install .[dev])
fi
