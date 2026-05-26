#!/usr/bin/env bash

# Copyright Contributors to the Pyro project.
#
# SPDX-License-Identifier: Apache-2.0

# visdom is distributed as an sdist whose setup.py imports pkg_resources.
# Modern setuptools (>=81) no longer provides pkg_resources in pip's isolated
# build environment, so preinstall visdom without build isolation.

set -xe

pip install 'setuptools<81'
pip install --no-build-isolation 'visdom>=0.2.3'
