#!/usr/bin/env bash

set -xe


TMP_DIR=tmp

function _cleanup() {
    [[ -d ${TMP_DIR} ]] && rm -rf ${TMP_DIR}
}

trap _cleanup EXIT

# Adjust as per the version used in CI.
PYTORCH_VERSION=0.4.0a0
PYTORCH_BUILD_COMMIT=$(grep 'git checkout .* a well-tested commit' README.md | cut -f3 -d' ')


# Detect OS and Python version numbers.
OS=$(uname -s)
PYTHON_VERSION=$(python -c 'import sys; version=sys.version_info[:3]; print("{0}{1}".format(*version))')

# Lookup wheel names to download
WHL_VERSION=${PYTORCH_VERSION}%2B${PYTORCH_BUILD_COMMIT}
PYTORCH_MAC_PY_27_WHL="torch-${WHL_VERSION}-cp27-cp27m-macosx_10_6_x86_64"
PYTORCH_MAC_PY_35_WHL="torch-${WHL_VERSION}-cp35-cp35m-macosx_10_6_x86_64"
PYTORCH_MAC_PY_36_WHL="torch-${WHL_VERSION}-cp36-cp36m-macosx_10_7_x86_64"
PYTORCH_LINUX_PY_27_WHL="torch-${WHL_VERSION}-cp27-cp27mu-linux_x86_64"
PYTORCH_LINUX_PY_35_WHL="torch-${WHL_VERSION}-cp35-cp35m-linux_x86_64"
PYTORCH_LINUX_PY_36_WHL="torch-${WHL_VERSION}-cp36-cp36m-linux_x86_64"

# Cloudfront path for the builds
PYTORCH_LINUX_PREFIX="https://d2fefpcigoriu7.cloudfront.net/pytorch-build/linux-cpu"
PYTORCH_OSX_PREFIX="https://d2fefpcigoriu7.cloudfront.net/pytorch-build/mac-cpu"


mkdir -p ${TMP_DIR}

if [[ ${OS} == Darwin* ]]; then
    WHL_PREFIX=${PYTORCH_OSX_PREFIX}
    WHL_LOOKUP=PYTORCH_MAC_PY_${PYTHON_VERSION}_WHL
elif [[ ${OS} == Linux* ]]; then
    WHL_PREFIX=${PYTORCH_LINUX_PREFIX}
    WHL_LOOKUP=PYTORCH_LINUX_PY_${PYTHON_VERSION}_WHL
else
    echo "OS - ${OS} is not supported by the install script."
    exit 1
fi

# Download wheel and install
curl -o tmp/"${!WHL_LOOKUP}.whl" "${WHL_PREFIX}/${!WHL_LOOKUP}.whl"

pip install tmp/*
