#!/usr/bin/env bash
set -x

CURRENT_HEAD=$(git rev-parse --abbrev-ref HEAD)
# If the current state is detached head, store current commit info instead.
if [ ${CURRENT_HEAD} = 'HEAD' ]; then
    git checkout -b tmp
    CURRENT_HEAD=$(git rev-parse HEAD)
fi
# Reference is with respect to the `dev` branch.
REF_HEAD=${1:-dev}
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
    pytest -vs tests/perf/test_benchmark.py --benchmark-save=dev --benchmark-save=${REF_HEAD}
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

# Run benchmark comparison - fails if the min run time is 10% less than on the ref branch.
if [ ${IS_BENCHMARK_FILE_IN_DEV} = 1 ]; then
    pytest -vx tests/perf/test_benchmark.py --benchmark-compare --benchmark-compare-fail=min:10% --name=short
else
    pytest -vx tests/perf/test_benchmark.py
fi
