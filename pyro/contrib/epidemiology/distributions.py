# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from contextlib import contextmanager

import torch

import pyro.distributions as dist

_APPROX_SAMPLE_THRESH = 10000


@contextmanager
def set_approx_sample_thresh(thresh):
    """
    Temporarily set global approx_sample_thresh in ``infection_dist``.
    The default global value is 10000.

    :param thresh: New temporary threshold.
    :type thresh: int or float.
    """
    global _APPROX_SAMPLE_THRESH
    old = _APPROX_SAMPLE_THRESH
    try:
        _APPROX_SAMPLE_THRESH = thresh
        yield
    finally:
        _APPROX_SAMPLE_THRESH = old


def infection_dist(*,
                   individual_rate,
                   num_infectious,
                   num_susceptible=math.inf,
                   population=math.inf,
                   concentration=math.inf,
                   approx_sample_thresh=None):
    """
    Create a :class:`~pyro.distributions.Distribution` over the number of new
    infections at a discrete time step.

    This returns a Poisson, Negative-Binomial, Binomial, or Beta-Binomial
    distribution depending on whether ``population`` and ``concentration`` are
    finite. In Pyro models, the population is usually finite. In the limit
    ``population → ∞`` and ``num_susceptible/population → 1``, the Binomial
    converges to Poisson and the Beta-Binomial converges to Negative-Binomial.
    In the limit ``concentration → ∞``, the Negative-Binomial converges to
    Poisson and the Beta-Binomial converges to Binomial.

    The overdispersed distributions (Negative-Binomial and Beta-Binomial
    returned when ``concentration < ∞``) are useful for modeling superspreader
    individuals [1,2]. The finitely supported distributions Binomial and
    Negative-Binomial are useful in small populations and in probabilistic
    programming systems where truncation or censoring are expensive [3].

    **References**

    [1] J. O. Lloyd-Smith, S. J. Schreiber, P. E. Kopp, W. M. Getz (2005)
        "Superspreading and the effect of individual variation on disease
        emergence"
        https://www.nature.com/articles/nature04153.pdf
    [2] Lucy M. Li, Nicholas C. Grassly, Christophe Fraser (2017)
        "Quantifying Transmission Heterogeneity Using Both Pathogen Phylogenies
        and Incidence Time Series"
        https://academic.oup.com/mbe/article/34/11/2982/3952784
    [3] Lawrence Murray et al. (2018)
        "Delayed Sampling and Automatic Rao-Blackwellization of Probabilistic
        Programs"
        https://arxiv.org/pdf/1708.07787.pdf

    :param individual_rate: The mean number of infections per infectious
        individual per time step in the limit of large population, equal to
        ``R0 / tau`` where ``R0`` is the basic reproductive number and ``tau``
        is the mean duration of infectiousness.
    :param num_infectious: The number of infectious individuals at this
        time step, sometimes ``I``, sometimes ``E+I``.
    :param num_susceptible: The number ``S`` of susceptible individuals at this
        time step. This defaults to an infinite population.
    :param population: The total number of individuals in a population.
        This defaults to an infinite population.
    :param concentration: The concentration or dispersion parameter ``k`` in
        overdispersed models of superspreaders [1,2]. This defaults to minimum
        variance ``concentration = ∞``.
    :param approx_sample_thresh: Population threshold above which Binomial
        samples will be approximated as clamped Poisson samples, including
        internally in BetaBinomial sampling. Defaults to the global value which
        defaults to 10000.
    """
    # Convert to colloquial variable names.
    R = individual_rate
    I = num_infectious
    S = num_susceptible
    N = population
    k = concentration

    if isinstance(N, float) and N == math.inf:
        if isinstance(k, float) and k == math.inf:
            # Return a Poisson distribution.
            return dist.Poisson(R * I)
        else:
            # Return an overdispersed Negative-Binomial distribution.
            combined_k = k * I
            logits = torch.as_tensor(R / k).log()
            return dist.NegativeBinomial(combined_k, logits=logits)
    else:
        # Compute the probability that any given (susceptible, infectious)
        # pair of individuals results in an infection at this time step.
        p = torch.as_tensor(R / N).clamp(max=1 - 1e-6)
        # Combine infections from all individuals.
        combined_p = p.neg().log1p().mul(I).expm1().neg()  # = 1 - (1 - p)**I
        combined_p = combined_p.clamp(min=1e-6)
        if approx_sample_thresh is None:
            approx_sample_thresh = _APPROX_SAMPLE_THRESH

        if isinstance(k, float) and k == math.inf:
            # Return a pure Binomial model, combining the independent Binomial
            # models of each infectious individual.
            return dist.ExtendedBinomial(
                S, combined_p, approx_sample_thresh=approx_sample_thresh)
        else:
            # Return an overdispersed Beta-Binomial model, combining
            # independent BetaBinomial(c1,c0,S) models for each infectious
            # individual.
            c1 = (k * I).clamp(min=1e-6)
            c0 = c1 * (combined_p.reciprocal() - 1).clamp(min=1e-6)
            return dist.ExtendedBetaBinomial(
                c1, c0, S, approx_sample_thresh=approx_sample_thresh)
