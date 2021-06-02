# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, Union

import torch

from pyro.ops.welford import WelfordCovariance


class StreamingStats(ABC):
    """
    Base class for streamable statistics of dictionaries of tensors.

    Derived classes must implelement :meth:`update`, :meth:`merge`, and
    :meth:`get`.
    """

    @abstractmethod
    def update(self, sample: Dict[str, torch.Tensor]) -> None:
        """
        Update state from a single sample.

        This mutates ``self`` and returns nothing. Updates should be
        independent of order, i.e. samples should be exchangeable.

        :param dict sample: A sample dictionary mapping sample name to sample
            value (a :class:`torch.Tensor`).
        """
        raise NotImplementedError

    @abstractmethod
    def merge(self, other: "StreamingStats") -> "StreamingStats":
        """
        Select two aggregate statistics, e.g. from different MCMC chains.

        This is a pure function: it returns a new :class:`StreamingStats`
        object and does not modify either ``self`` or ``other``.

        :param other: Another streaming stats instance of the same type.
        """
        assert isinstance(other, type(self))
        raise NotImplementedError

    @abstractmethod
    def get(self) -> Dict[str, object]:
        """
        Return the aggregate statistic. This is typically a tensor or
        collection of tensors, but can be any Python object.
        """
        raise NotImplementedError


class SelectStats(StreamingStats):
    """
    Collection to select different statistics for each sample name.

    This is useful for computing statistics for only a subset of sample names,
    or computing different statistics for different samples (e.g. expensive
    statistics for small tensors and cheap statistis for large tensors).

    :param stats: A dictionary mapping sample name to :class:`StreamingStats`
        instance corresponding to that name. Note instances can be reused by
        multiple keys.
    """
    def __init__(self, stats: Dict[object, StreamingStats]):
        self.stats = stats

    def update(self, sample: Dict[str, torch.Tensor]) -> None:
        assert isinstance(sample, dict)
        for k, stat in self.stats.items():
            stat.update({k: sample[k]})

    def merge(self, other: "SelectStats") -> "SelectStats":
        assert isinstance(other, type(self))
        assert set(self.stats) == set(other.stats)
        merged: Dict[int, SelectStats] = {}
        result: Dict[str, SelectStats] = {}
        for key, lhs in self.stats.items():
            rhs = other.stats[key]
            ids = id(lhs), id(rhs)
            if ids in merged:
                stat = merged[ids]
            else:
                merged[ids] = stat = lhs.merge(rhs)
            result[key] = stat
        return type(self)(result)

    def get(self) -> Dict[str, object]:
        result = {}
        for k, v in self.stats.items():
            result.update(v.get())
        return result


class DictStats(StreamingStats):
    """
    Collection of multiple streaming statistics into a dictionary.

    This is useful for computing multiple statistics for each sample.

    :param stats: A dictionary mapping statistic name to
        :class:`StreamingStats` instance corresponding to that name.
    """

    def __init__(self, stats: Dict[str, StreamingStats]):
        self.stats = stats

    def update(self, sample: Dict[str, torch.Tensor]) -> None:
        assert isinstance(sample, dict)
        for stat in self.stats.values():
            stat.update(sample)

    def merge(self, other: "DictStats") -> "DictStats":
        assert isinstance(other, type(self))
        assert set(other.stats) == set(self.stats)
        stats = {
            name: stat.merge(other.stats[name])
            for name, stat in self.stats.items()
        }
        return type(self)(stats)

    def get(self) -> Dict[str, Dict[str, object]]:
        result = defaultdict(dict)
        for name, stat in self.stats.items():
            for k, v in stat.get().items():
                result[k][name] = v
        return dict(result)


class CountMeanStats(StreamingStats):
    """
    Statistic tracking the count and mean of all sample tensors.
    """

    def __init__(self):
        self.count = defaultdict(int)
        self.mean = {}

    def update(self, sample: Dict[str, torch.Tensor]):
        assert isinstance(sample, dict)
        for k, v in sample.items():
            self.count[k] += 1
            if self.count[k] == 1:
                self.mean[k] = v.detach().clone()
            else:
                self.mean[k] += (v.detach() - self.mean[k]) / self.count[k]

    def merge(self, other: "CountMeanStats") -> "CountMeanStats":
        assert isinstance(other, type(self))
        keys = set(self.count).union(other.count)
        result = CountMeanStats()
        for k in keys:
            if self.count.get(k, 0) == 0:
                result.count[k] = other.count[k]
                result.mean[k] = other.mean[k].clone()
            elif other.count.get(k, 0) == 0:
                result.count[k] = self.count[k]
                result.mean[k] = self.mean[k].clone()
            else:
                result.count[k] = self.count[k] + other.count[k]
                result.mean[k] = (self.mean[k] * (self.count[k] / result.count[k])
                                  + other.mean[k] * (other.count[k] / result.count[k]))
        return result

    def get(self) -> Dict[str, Dict[str, Union[int, torch.Tensor]]]:
        return {k: {"count": self.count, "mean": v} for k, v in self.mean.items()}


class CountMeanVarianceStats(StreamingStats):
    """
    Statistic tracking the count, mean, and (diagonal) variance of all sample
    tensors.

    Note the count is shared across all samples.
    """
    def __init__(self):
        self.shape = {}
        self.welford = {}

    def update(self, sample: Dict[str, torch.Tensor]):
        assert isinstance(sample, dict)
        for k, v in sample.items():
            if k not in self.shape:
                self.shape[k] = v.shape
                self.welford[k] = WelfordCovariance(diagonal=True)
            assert v.shape == self.shape[k]
            self.welford[k].update(v.detach().reshape(-1))

    def merge(self, other: "CountMeanVarianceStats") -> "CountMeanVarianceStats":
        assert isinstance(other, type(self))
        keys = set(self.shape).union(other.shape)
        result = copy.deepcopy(self)
        for key in keys:
            if key not in self.shape:
                result.shape[key] = other.shape[key]
                result.welford[key] = copy.deepcopy(other.welford[key])
            elif key in other.shape:
                assert other.shape[key] == self.shape[key]
                lhs = self.welford[key]
                rhs = other.welford[key]
                res = result.welford[key]
                res.n_samples = lhs.n_samples + rhs.n_samples
                lhs_weight = lhs.n_samples / res.n_samples
                rhs_weight = rhs.n_samples / res.n_samples
                res._mean = lhs_weight * lhs._mean + rhs_weight * rhs._mean
                res._m2 = (
                    lhs._m2 + rhs._m2
                    + (lhs.n_samples * rhs.n_samples / res.n_samples)
                    * (lhs._mean - rhs._mean) ** 2
                )
        return result

    def get(self) -> Dict[str, Dict[str, Union[int, torch.Tensor]]]:
        result = {}
        for k, welford in self.welford.items():
            shape = self.shape[k]
            result[k] = {
                "count": welford.n_samples,
                "mean": welford._mean.reshape(shape),
                "variance": welford.get_covariance(regularize=False).reshape(shape),
            }
        return result


__all__ = [
    "StreamingStats",
    "SelectStats",
    "DictStats",
    "CountMeanStats",
    "CountMeanVarianceStats",
]
