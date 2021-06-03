# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import functools

import pytest
import torch

from pyro.ops.streaming import (
    CountMeanStats,
    CountMeanVarianceStats,
    CountStats,
    StackStats,
    StatsOfDict,
)
from tests.common import assert_close


def generate_data(num_samples):
    shapes = {"aaa": (), "bbb": (4,), "ccc": (3, 2), "ddd": (5, 1)}
    return [{k: torch.randn(v) for k, v in shapes.items()} for _ in range(num_samples)]


EXAMPLE_STATS = [
    CountStats,
    functools.partial(StatsOfDict, default=CountMeanStats),
    functools.partial(StatsOfDict, default=CountMeanVarianceStats),
    functools.partial(StatsOfDict, default=StackStats),
    StatsOfDict,
    functools.partial(
        StatsOfDict, {"aaa": CountMeanStats, "bbb": CountMeanVarianceStats}
    ),
]
EXAMPLE_STATS_IDS = [
    "CountStats",
    "CountMeanStats",
    "CountMeanVarianceStats",
    "StackStats",
    "StatsOfDict1",
    "StatsOfDict2",
]


def sort_samples_in_place(x):
    for key, value in list(x.items()):
        if isinstance(key, str) and key == "samples":
            x[key] = value.sort(0).values
        elif isinstance(value, dict):
            sort_samples_in_place(value)


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

    # Sort samples in case of StackStats.
    sort_samples_in_place(expected)
    sort_samples_in_place(actual)

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


def test_stats_of_dict():
    stats = StatsOfDict(types={"aaa": CountMeanStats}, default=CountStats)
    stats.update({"aaa": torch.tensor(0.0)})
    stats.update({"aaa": torch.tensor(1.0), "bbb": torch.randn(3, 3)})
    stats.update({"aaa": torch.tensor(2.0), "bbb": torch.randn(3, 3)})
    actual = stats.get()

    expected = {
        "aaa": {"count": 3, "mean": torch.tensor(1.0)},
        "bbb": {"count": 2},
    }
    assert_close(actual, expected)
