# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
# Copyright (c) 2015, Gamelan Labs, Inc.
# Copyright (c) 2016, Google, Inc.
# Copyright (c) 2019, Gamalon, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import division
from collections import defaultdict
from math import gamma
try:
    from itertools import izip as zip
except ImportError:
    pass
import random
import sys

import numpy
import numpy.random
from numpy import pi
from scipy.spatial import cKDTree

from .utils import chi2sf

NoneType = type(None)

#: Data types for integral random variables.
#:
#: For Python 2.7, this tuple also includes `long`.
INTEGRAL_TYPES = (int, )

#: Data types for continuous random variables.
CONTINUOUS_TYPES = (float, numpy.float32, numpy.float64)

#: Data types for discrete random variables.
#:
#: For Python 2.7, this tuple also includes `long` and `basestring`.
DISCRETE_TYPES = (NoneType, bool, int, str, numpy.int32, numpy.int64)

if sys.version_info < (3, ):
    # `str` is a subclass of `basestring`, so this is doing a little
    # more work than is necessary, but it should not cause a problem.
    DISCRETE_TYPES += (long, basestring)  # noqa
    INTEGRAL_TYPES += (long, )  # noqa


def seed_all(seed):
    random.seed(seed)
    numpy.random.seed(seed)


def get_dim(thing):
    if hasattr(thing, '__len__'):
        return len(thing)
    else:
        return 1


def print_histogram(probs, counts):
    WIDTH = 60.0
    max_count = max(counts)
    print('{: >8} {: >8}'.format('Prob', 'Count'))
    for prob, count in sorted(zip(probs, counts), reverse=True):
        width = int(round(WIDTH * count / max_count))
        print('{: >8.3f} {: >8d} {}'.format(prob, count, '-' * width))


def multinomial_goodness_of_fit(
        probs,
        counts,
        total_count,
        truncated=False,
        plot=False):
    """
    Pearson's chi^2 test, on possibly truncated data.
    http://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test

    Returns:
        p-value of truncated multinomial sample.
    """
    assert len(probs) == len(counts)
    assert truncated or total_count == sum(counts)
    chi_squared = 0
    dof = 0
    if plot:
        print_histogram(probs, counts)
    for p, c in zip(probs, counts):
        if abs(p - 1) < 1e-8:
            return 1 if c == total_count else 0
        assert p < 1, 'bad probability: %g' % p
        if p > 0:
            mean = total_count * p
            variance = total_count * p * (1 - p)
            assert variance > 1,\
                'WARNING goodness of fit is inaccurate; use more samples'
            chi_squared += (c - mean) ** 2 / variance
            dof += 1
        else:
            print('WARNING zero probability in goodness-of-fit test')
            if c > 0:
                return float('inf')

    if not truncated:
        dof -= 1

    survival = chi2sf(chi_squared, dof)
    return survival


def unif01_goodness_of_fit(samples, plot=False):
    """
    Bin uniformly distributed samples and apply Pearson's chi^2 test.
    """
    samples = numpy.array(samples, dtype=float)
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0
    bin_count = int(round(len(samples) ** 0.333))
    assert bin_count >= 7, 'WARNING imprecise test, use more samples'
    probs = numpy.ones(bin_count, dtype=numpy.float) / bin_count
    counts = numpy.zeros(bin_count, dtype=numpy.int)
    for sample in samples:
        counts[min(bin_count - 1, int(bin_count * sample))] += 1
    return multinomial_goodness_of_fit(probs, counts, len(samples), plot=plot)


def exp_goodness_of_fit(
        samples,
        plot=False,
        normalized=True,
        return_dict=False):
    """
    Transform exponentially distribued samples to unif01 distribution
    and assess goodness of fit via binned Pearson's chi^2 test.

    Inputs:
        samples - a list of real-valued samples from a candidate distribution
    """
    result = {}
    if not normalized:
        result['norm'] = numpy.mean(samples)
        samples /= result['norm']
    unif01_samples = numpy.exp(-samples)
    result['gof'] = unif01_goodness_of_fit(unif01_samples, plot=plot)
    return result if return_dict else result['gof']


def density_goodness_of_fit(
        samples,
        probs,
        plot=False,
        normalized=True,
        return_dict=False):
    """
    Transform arbitrary continuous samples to unif01 distribution
    and assess goodness of fit via binned Pearson's chi^2 test.

    Inputs:
        samples - a list of real-valued samples from a distribution
        probs - a list of probability densities evaluated at those samples
    """
    assert len(samples) == len(probs)
    assert len(samples) > 100, 'WARNING imprecision; use more samples'
    pairs = list(zip(samples, probs))
    pairs.sort()
    samples = numpy.array([x for x, p in pairs])
    probs = numpy.array([p for x, p in pairs])
    density = len(samples) * numpy.sqrt(probs[1:] * probs[:-1])
    gaps = samples[1:] - samples[:-1]
    exp_samples = density * gaps
    return exp_goodness_of_fit(exp_samples, plot, normalized, return_dict)


def volume_of_sphere(dim, radius):
    assert isinstance(dim, INTEGRAL_TYPES)
    return radius ** dim * pi ** (0.5 * dim) / gamma(0.5 * dim + 1)


def get_nearest_neighbor_distances(samples):
    if not hasattr(samples[0], '__iter__'):
        samples = numpy.array([samples]).T
    distances, indices = cKDTree(samples).query(samples, k=2)
    return distances[:, 1]


def vector_density_goodness_of_fit(
        samples,
        probs,
        plot=False,
        normalized=True,
        return_dict=False):
    """
    Transform arbitrary multivariate continuous samples
    to unif01 distribution via nearest neighbor distribution [1,2,3]
    and assess goodness of fit via binned Pearson's chi^2 test.

    [1] http://projecteuclid.org/download/pdf_1/euclid.aop/1176993668
    [2] http://arxiv.org/pdf/1006.3019v2.pdf
    [3] http://en.wikipedia.org/wiki/Nearest_neighbour_distribution

    Inputs:
        samples - a list of real-vector-valued samples from a distribution
        probs - a list of probability densities evaluated at those samples
    """
    assert len(samples)
    assert len(samples) == len(probs)
    dim = get_dim(samples[0])
    assert dim
    assert len(samples) > 1000 * dim, 'WARNING imprecision; use more samples'
    radii = get_nearest_neighbor_distances(samples)
    density = len(samples) * numpy.array(probs)
    volume = volume_of_sphere(dim, radii)
    exp_samples = density * volume
    return exp_goodness_of_fit(exp_samples, plot, normalized, return_dict)


def trivial_density_goodness_of_fit(
        samples,
        probs,
        plot=False,
        normalized=True,
        return_dict=False):
    assert len(samples)
    assert all(sample == samples[0] for sample in samples)
    result = {'gof': 1.0}
    if not normalized:
        result['norm'] = probs[0]
    if return_dict:
        return result
    else:
        return result['gof']


def auto_density_goodness_of_fit(
        samples,
        probs,
        plot=False,
        normalized=True,
        return_dict=False):
    """
    Dispatch on sample dimention and delegate to one of:
    - density_goodness_of_fit
    - vector_density_goodness_of_fit
    - trivial_density_goodness_of_fit
    """
    assert len(samples)
    dim = get_dim(samples[0])
    if dim == 0:
        fun = trivial_density_goodness_of_fit
    elif dim == 1:
        fun = density_goodness_of_fit
        if hasattr(samples[0], '__len__'):
            samples = [sample[0] for sample in samples]
    else:
        fun = vector_density_goodness_of_fit
    return fun(samples, probs, plot, normalized, return_dict)


def discrete_goodness_of_fit(
        samples,
        probs_dict,
        truncate_beyond=8,
        plot=False,
        normalized=True):
    """
    Transform arbitrary discrete data to multinomial
    and assess goodness of fit via Pearson's chi^2 test.
    """
    if not normalized:
        norm = sum(probs_dict.values())
        probs_dict = {i: p / norm for i, p in probs_dict.items()}
    counts = defaultdict(lambda: 0)
    for sample in samples:
        assert sample in probs_dict
        counts[sample] += 1
    items = [(prob, counts.get(i, 0)) for i, prob in probs_dict.items()]
    items.sort(reverse=True)
    truncated = (truncate_beyond and truncate_beyond < len(items))
    if truncated:
        items = items[:truncate_beyond]
    probs = [prob for prob, count in items]
    counts = [count for prob, count in items]
    assert sum(counts) > 100, 'WARNING imprecision; use more samples'
    return multinomial_goodness_of_fit(
        probs,
        counts,
        len(samples),
        truncated=truncated,
        plot=plot)


def split_discrete_continuous(data):
    """
    Convert arbitrary data to a pair `(discrete, continuous)`
    where `discrete` is hashable and `continuous` is a list of floats.
    """
    if isinstance(data, DISCRETE_TYPES):
        return data, []
    elif isinstance(data, CONTINUOUS_TYPES):
        return None, [data]
    elif isinstance(data, (tuple, list)):
        discrete = []
        continuous = []
        for part in data:
            d, c = split_discrete_continuous(part)
            discrete.append(d)
            continuous += c
        return tuple(discrete), continuous
    elif isinstance(data, numpy.ndarray):
        assert data.dtype in [numpy.float64, numpy.float32]
        return (None,) * len(data), list(map(float, data))
    else:
        raise TypeError(
            'split_discrete_continuous does not accept {} of type {}'.format(
                repr(data), str(type(data))))


def mixed_density_goodness_of_fit(samples, probs, plot=False, normalized=True):
    """
    Test general mixed discrete+continuous datatypes by
    (1) testing the continuous part conditioned on each discrete value
    (2) testing the discrete part marginalizing over the continuous part
    (3) testing the estimated total probability (if normalized = True)

    Inputs:
        samples - a list of plain-old-data samples from a distribution
        probs - a list of probability densities evaluated at those samples
    """
    assert len(samples)
    discrete_samples = []
    strata = defaultdict(lambda: ([], []))
    for sample, prob in zip(samples, probs):
        d, c = split_discrete_continuous(sample)
        discrete_samples.append(d)
        samples, probs = strata[d]
        samples.append(c)
        probs.append(prob)

    # Continuous part
    gofs = []
    discrete_probs = {}
    for key, (samples, probs) in strata.items():
        result = auto_density_goodness_of_fit(
            samples,
            probs,
            plot=plot,
            normalized=False,
            return_dict=True)
        gofs.append(result['gof'])
        discrete_probs[key] = result['norm']

    # Discrete part
    if len(strata) > 1:
        gofs.append(discrete_goodness_of_fit(
            discrete_samples,
            discrete_probs,
            plot=plot,
            normalized=False))

    # Normalization
    if normalized:
        norm = sum(discrete_probs.values())
        discrete_counts = [len(samples) for samples, _ in strata.values()]
        norm_variance = sum(1.0 / count for count in discrete_counts)
        dof = len(discrete_counts)
        chi_squared = (1 - norm) ** 2 / norm_variance
        gofs.append(chi2sf(chi_squared, dof))
        if plot:
            print('norm = {:.4g} +- {:.4g}'.format(norm, norm_variance ** 0.5))
            print('     = {}'.format(
                ' + '.join(map('{:.4g}'.format, discrete_probs.values()))))

    return min(gofs)

