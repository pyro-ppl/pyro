# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, Dict, Hashable, Union

import torch

from pyro.ops.welford import WelfordCovariance


class StreamingStats(ABC):
    """
    Abstract base class for streamable statistics of trees of tensors.

    Derived classes must implelement :meth:`update`, :meth:`merge`, and
    :meth:`get`.
    """

    @abstractmethod
    def update(self, sample) -> None:
        """
        Update state from a single sample.

        This mutates ``self`` and returns nothing. Updates should be
        independent of order, i.e. samples should be exchangeable.

        :param sample: A sample value which is a nested dictionary of
            :class:`torch.Tensor` leaves. This can have arbitrary nesting and
            shape shape, but assumes shape is constant across calls to
            ``.update()``.
        """
        raise NotImplementedError

    @abstractmethod
    def merge(self, other) -> "StreamingStats":
        """
        Select two aggregate statistics, e.g. from different MCMC chains.

        This is a pure function: it returns a new :class:`StreamingStats`
        object and does not modify either ``self`` or ``other``.

        :param other: Another streaming stats instance of the same type.
        """
        assert isinstance(other, type(self))
        raise NotImplementedError

    @abstractmethod
    def get(self) -> Any:
        """
        Return the aggregate statistic.
        """
        raise NotImplementedError


class CountStats(StreamingStats):
    """
    Statistic tracking only the number of samples.

    For example::

        >>> stats = CountStats()
        >>> stats.update(torch.randn(3, 3))
        >>> stats.get()
        {'count': 1}
    """

    def __init__(self):
        self.count = 0
        super().__init__()

    def update(self, sample) -> None:
        self.count += 1

    def merge(self, other: "CountStats") -> "CountStats":
        assert isinstance(other, type(self))
        result = CountStats()
        result.count = self.count + other.count
        return result

    def get(self) -> Dict[str, int]:
        """
        :returns: A dictionary with keys ``count: int``.
        :rtype: dict
        """
        return {"count": self.count}


class StatsOfDict(StreamingStats):
    """
    Statistics of samples that are dictionaries with constant set of keys.

    For example the following are equivalent::

        # Version 1. Hand encode statistics.
        >>> a_stats = CountStats()
        >>> b_stats = CountMeanStats()
        >>> a_stats.update(torch.tensor(0.))
        >>> b_stats.update(torch.tensor([1., 2.]))
        >>> summary = {"a": a_stats.get(), "b": b_stats.get()}

        # Version 2. Collect samples into dictionaries.
        >>> stats = StatsOfDict({"a": CountStats, "b": CountMeanStats})
        >>> stats.update({"a": torch.tensor(0.), "b": torch.tensor([1., 2.])})
        >>> summary = stats.get()
        >>> summary
        {'a': {'count': 1}, 'b': {'count': 1, 'mean': tensor([1., 2.])}}

    :param default: Default type of statistics of values of the dictionary.
        Defaults to the inexpensive :class:`CountStats`.
    :param dict types: Dictionary mapping key to type of statistic that should
        be recorded for values corresponding to that key.
    """

    def __init__(
        self,
        types: Dict[Hashable, Callable[[], StreamingStats]] = {},
        default: Callable[[], StreamingStats] = CountStats,
    ):
        self.stats: Dict[Hashable, StreamingStats] = defaultdict(default)
        self.stats.update({k: v() for k, v in types.items()})
        super().__init__()

    def update(self, sample: Dict[Hashable, Any]) -> None:
        for k, v in sample.items():
            self.stats[k].update(v)

    def merge(self, other: "StatsOfDict") -> "StatsOfDict":
        assert isinstance(other, type(self))
        result = copy.deepcopy(self)
        for k in set(self.stats).union(other.stats):
            if k not in self.stats:
                result.stats[k] = copy.deepcopy(other.stats[k])
            elif k in other.stats:
                result.stats[k] = self.stats[k].merge(other.stats[k])
        return result

    def get(self) -> Dict[Hashable, Any]:
        """
        :returns: A dictionary of statistics. The keys of this dictionary are
            the same as the keys of the samples from which this object is
            updated.
        :rtype: dict
        """
        return {k: v.get() for k, v in self.stats.items()}


class StackStats(StreamingStats):
    """
    Statistic collecting a stream of tensors into a single stacked tensor.
    """

    def __init__(self):
        self.samples = []

    def update(self, sample: torch.Tensor) -> None:
        assert isinstance(sample, torch.Tensor)
        self.samples.append(sample)

    def merge(self, other: "StackStats") -> "StackStats":
        assert isinstance(other, type(self))
        result = StackStats()
        result.samples = self.samples + other.samples
        return result

    def get(self) -> Dict[str, Union[int, torch.Tensor]]:
        """
        :returns: A dictionary with keys ``count: int`` and (if any samples
            have been collected) ``samples: torch.Tensor``.
        :rtype: dict
        """
        if not self.samples:
            return {"count": 0}
        return {"count": len(self.samples), "samples": torch.stack(self.samples)}


class CountMeanStats(StreamingStats):
    """
    Statistic tracking the count and mean of a single :class:`torch.Tensor`.
    """

    def __init__(self):
        self.count = 0
        self.mean = 0
        super().__init__()

    def update(self, sample: torch.Tensor) -> None:
        assert isinstance(sample, torch.Tensor)
        self.count += 1
        self.mean += (sample.detach() - self.mean) / self.count

    def merge(self, other: "CountMeanStats") -> "CountMeanStats":
        assert isinstance(other, type(self))
        result = CountMeanStats()
        result.count = self.count + other.count
        p = self.count / max(result.count, 1)
        q = other.count / max(result.count, 1)
        result.mean = p * self.mean + q * other.mean
        return result

    def get(self) -> Dict[str, Union[int, torch.Tensor]]:
        """
        :returns: A dictionary with keys ``count: int`` and (if any samples
            have been collected) ``mean: torch.Tensor``.
        :rtype: dict
        """
        if self.count == 0:
            return {"count": 0}
        return {"count": self.count, "mean": self.mean}


class CountMeanVarianceStats(StreamingStats):
    """
    Statistic tracking the count, mean, and (diagonal) variance of a single
    :class:`torch.Tensor`.
    """

    def __init__(self):
        self.shape = None
        self.welford = WelfordCovariance(diagonal=True)
        super().__init__()

    def update(self, sample: torch.Tensor) -> None:
        assert isinstance(sample, torch.Tensor)
        if self.shape is None:
            self.shape = sample.shape
        assert sample.shape == self.shape
        self.welford.update(sample.detach().reshape(-1))

    def merge(self, other: "CountMeanVarianceStats") -> "CountMeanVarianceStats":
        assert isinstance(other, type(self))
        if self.shape is None:
            return copy.deepcopy(other)
        if other.shape is None:
            return copy.deepcopy(self)
        result = copy.deepcopy(self)
        res = result.welford
        lhs = self.welford
        rhs = other.welford
        res.n_samples = lhs.n_samples + rhs.n_samples
        lhs_weight = lhs.n_samples / res.n_samples
        rhs_weight = rhs.n_samples / res.n_samples
        res._mean = lhs_weight * lhs._mean + rhs_weight * rhs._mean
        res._m2 = (
            lhs._m2
            + rhs._m2
            + (lhs.n_samples * rhs.n_samples / res.n_samples)
            * (lhs._mean - rhs._mean) ** 2
        )
        return result

    def get(self) -> Dict[str, Union[int, torch.Tensor]]:
        """
        :returns: A dictionary with keys ``count: int`` and (if any samples
            have been collected) ``mean: torch.Tensor`` and ``variance:
            torch.Tensor``.
        :rtype: dict
        """
        if self.shape is None:
            return {"count": 0}
        count = self.welford.n_samples
        mean = self.welford._mean.reshape(self.shape)
        variance = self.welford.get_covariance(regularize=False).reshape(self.shape)
        return {"count": count, "mean": mean, "variance": variance}


# Note this is ordered logically for sphinx rather than alphabetically.
__all__ = [
    "StreamingStats",
    "StatsOfDict",
    "StackStats",
    "CountStats",
    "CountMeanStats",
    "CountMeanVarianceStats",
]
