#!/usr/bin/env bash

set -xe

function _cleanup() {
    [[ ${#DIRSTACK[@]} -gt 1 ]] && popd
    [[ -d ${REF_TMP_DIR} ]] && rm -rf ${REF_TMP_DIR}
}

trap _cleanup EXIT

# Reference is with respect to the `dev` branch, by default.
REF_HEAD=${1:-dev}
REF_COMMIT=$(git rev-parse --short HEAD)
BENCHMARK_FILE=tests/perf/test_benchmark.py
BENCHMARK_DIR=$( cd $(dirname "$0")/../.benchmarks ; pwd -P )
REF_TMP_DIR=.tmp_test_dir

CURRENT_HEAD=$(git rev-parse --abbrev-ref HEAD)
CURRENT_COMMIT=$(git rev-parse --short HEAD)
# If the current state is detached head, store current commit info instead.
if [ ${CURRENT_HEAD} = 'HEAD' ]; then
    CURRENT_HEAD=$(git rev-parse HEAD)
fi

# clone the repo into the temporary directory and run benchmark tests
git clone -b ${REF_HEAD} --single-branch . ${REF_TMP_DIR}
pushd ${REF_TMP_DIR}

# Skip if benchmark utils are not on `dev` branch.
if [ -e ${BENCHMARK_FILE} ]; then
    python tests/perf/test_benchmark.py --models "${@:2}" --suffix ${REF_HEAD}${REF_COMMIT} \
        --benchmark_dir ${BENCHMARK_DIR} || echo "ERR: Failed on branch upstream/${REF_HEAD}."
fi

# cd back into the current repo to run comparison benchmarks
popd

# Run profiling on current commit
python tests/perf/test_benchmark.py --models "${@:2}" --suffix ${CURRENT_HEAD}${CURRENT_COMMIT}

set +x

for filename in .benchmarks/*.prof; do
    for model in "${@:2}"; do
        # Open the two profiles in snakeviz
        if [[ "${filename}" =~ .*${model}.*(${CURRENT_COMMIT}.prof|${REF_COMMIT}).prof ]]; then
            snakeviz ${filename} &>/dev/null &
        fi
    done
done
