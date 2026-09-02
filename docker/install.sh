#!/usr/bin/env bash

# Copyright Contributors to the Pyro project.
#
# SPDX-License-Identifier: Apache-2.0

set -xe

pip install --upgrade pip
pip install notebook ipywidgets matplotlib

# 1. Install PyTorch
# Use conda package if pytorch_branch = 'release'.
# Else, install from source, using git branch `pytorch_branch`

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/${pytorch_whl}
if [ ${pytorch_branch} != "release" ]
then
    git clone --recursive https://github.com/pytorch/pytorch.git
    pushd pytorch && git checkout ${pytorch_branch}
    pip uninstall -y torch
    conda install cmake ninja
    pip install -r requirements.txt
    pip install mkl-static mkl-include
    if [ ${pytorch_whl} != "cpu" ]
    then
        conda install -c pytorch magma-cuda${pytorch_whl:2}
    fi
    pip install -e .
    popd
fi


# 2. Install Pyro
# Use pypi wheel if pyro_branch = 'release'.
# Else, install from source, using git branch `pyro_branch`
if [ ${pyro_branch} = "release" ]
then
    pip install pyro-ppl
else
    git clone ${pyro_git_url}
    (cd pyro && git checkout ${pyro_branch} && pip install -e .[dev])
fi
