# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.ops.streaming import (
    CountMeanStats,
    CountMeanVarianceStats,
    DictStats,
    SelectStats,
)
from tests.common import assert_close


def generate_data(num_samples):
    shapes = {"aaa": (), "bbb": (4,), "ccc": (3, 2), "ddd": (5, 1)}
    return [
        {k: torch.randn(v) for k, v in shapes.items()}
        for _ in range(num_samples)
    ]


EXAMPLE_STATS = [
    lambda: CountMeanStats(),
    lambda: CountMeanVarianceStats(),
    lambda: SelectStats({"aaa": CountMeanVarianceStats(), "bbb": CountMeanStats()}),
    lambda: DictStats({"cm": CountMeanStats(), "cmv": CountMeanVarianceStats()}),
]
EXAMPLE_STATS_IDS = [type(x()).__name__ for x in EXAMPLE_STATS]


@pytest.mark.parametrize("size", [0, 10])
@pytest.mark.parametrize("make_stats", EXAMPLE_STATS, ids=EXAMPLE_STATS_IDS)
def test_update_get(make_stats, size):
    samples = generate_data(size)

    expected_stats = make_stats()
    for sample in samples:
        expected_stats.update(sample)
    expected = expected_stats.get()

    actual_stats = make_stats()
    for i in torch.randperm(len(samples)).tolist():
        actual_stats.update(samples[i])
    actual = actual_stats.get()

    assert_close(actual, expected)


@pytest.mark.parametrize("left_size, right_size", [(3, 5), (0, 8), (8, 0), (0, 0)])
@pytest.mark.parametrize("make_stats", EXAMPLE_STATS, ids=EXAMPLE_STATS_IDS)
def test_update_merge_get(make_stats, left_size, right_size):
    left_samples = generate_data(left_size)
    right_samples = generate_data(right_size)

    expected_stats = make_stats()
    for sample in left_samples + right_samples:
        expected_stats.update(sample)
    expected = expected_stats.get()

    left_stats = make_stats()
    for sample in left_samples:
        left_stats.update(sample)
    right_stats = make_stats()
    for sample in right_samples:
        right_stats.update(sample)
    actual_stats = left_stats.merge(right_stats)
    assert isinstance(actual_stats, type(expected_stats))

    actual = actual_stats.get()
    assert_close(actual, expected)
