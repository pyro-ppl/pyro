#!/usr/bin/env bash

set -xe

function _cleanup() {
    [[ ${#DIRSTACK[@]} -gt 1 ]] && popd
    [[ -d ${REF_TMP_DIR} ]] && rm -rf ${REF_TMP_DIR}
}

trap _cleanup EXIT

# Reference is with respect to the `dev` branch, by default.
REF_HEAD=${1:-dev}
BENCHMARK_FILE=tests/perf/test_benchmark.py
IS_BENCHMARK_FILE_IN_DEV=1
REF_TMP_DIR=.tmp_test_dir

CURRENT_HEAD=$(git rev-parse --abbrev-ref HEAD)
# If the current state is detached head (e.g. travis), store current commit info instead.
if [ ${CURRENT_HEAD} = 'HEAD' ]; then
    CURRENT_HEAD=$(git rev-parse HEAD)
fi

# clone the repo into the temporary directory and run benchmark tests
git clone -b ${REF_HEAD} --single-branch . ${REF_TMP_DIR}
pushd ${REF_TMP_DIR}

# Skip if benchmark utils are not on `dev` branch.
if [ -e ${BENCHMARK_FILE} ]; then
    pytest -vs tests/perf/test_benchmark.py --benchmark-save=${REF_HEAD} --benchmark-name=short \
        --benchmark-columns=min,median,max --benchmark-sort=name \
        --benchmark-storage=file://../.benchmarks || echo "ERR: Failed on branch upstream/${REF_HEAD}." \
        --benchmark-timer timeit.default_timer
fi

# cd back into the current repo to run comparison benchmarks
popd

# Run benchmark comparison - fails if the min run time is 10% less than on the ref branch.
if [ ${IS_BENCHMARK_FILE_IN_DEV} = 1 ]; then
    pytest -vx tests/perf/test_benchmark.py --benchmark-compare --benchmark-compare-fail=min:15% \
        --benchmark-name=short --benchmark-columns=min,median,max --benchmark-sort=name \
        --benchmark-timer timeit.default_timer
else
    pytest -vx tests/perf/test_benchmark.py --benchmark-name=short --benchmark-columns=min,median,max \
        --benchmark-sort=name --benchmark-timer timeit.default_timer
fi
