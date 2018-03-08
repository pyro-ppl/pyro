#!/usr/bin/env bash
set -x

CURRENT_HEAD=$(git rev-parse --abbrev-ref HEAD)
CURRENT_COMMIT=$(git rev-parse --short HEAD)
# If the current state is detached head, store current commit info instead.
if [ ${CURRENT_HEAD} = 'HEAD' ]; then
    git checkout -b tmp
    CURRENT_HEAD=$(git rev-parse HEAD)
fi
# Reference is with respect to the `dev` branch.
REF_HEAD=${1:-dev}
REF_COMMIT=$(git rev-parse --short HEAD)
BENCHMARK_FILE=tests/perf/test_benchmark.py
IS_BENCHMARK_FILE_IN_DEV=1
REF_EXIT_CODE=0
IS_DIRTY=0

# If uncommitted changes exist, stash these and set a flag.
git diff-index --quiet HEAD -- || IS_DIRTY=1
git stash
git checkout ${REF_HEAD}

# Skip if benchmark utils are not on `dev` branch.
if [ -e ${BENCHMARK_FILE} ]; then
    python tests/perf/test_benchmark.py --models "${@:2}" --suffix ${REF_HEAD}${REF_COMMIT}
    REF_EXIT_CODE=$?
else
    IS_BENCHMARK_FILE_IN_DEV=0
fi

# Check out the initial git state and pop stash to get uncommitted changes back
git checkout ${CURRENT_HEAD}
if [ ${IS_DIRTY} = 1 ]; then
    git stash pop
fi

if [ ${REF_EXIT_CODE} != 0 ]; then
    echo 'Failure on reference branch.'
    exit 1
fi

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
