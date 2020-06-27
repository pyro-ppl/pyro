# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math
from contextlib import contextmanager

import torch
from torch.distributions.utils import broadcast_all

import pyro.distributions as dist
from pyro.distributions.util import is_validation_enabled

_RELAX = False
_RELAX_MIN_VARIANCE = 0.1


def _all(x):
    return x.all() if isinstance(x, torch.Tensor) else x


def _is_zero(x):
    return _all(x == 0)


@contextmanager
def set_approx_sample_thresh(thresh):
    """
    EXPERIMENTAL Context manager / decorator to temporarily set the global
    default value of ``Binomial.approx_sample_thresh``, thereby decreasing the
    computational complexity of sampling from
    :class:`~pyro.distributions.Binomial`,
    :class:`~pyro.distributions.BetaBinomial`,
    :class:`~pyro.distributions.ExtendedBinomial`,
    :class:`~pyro.distributions.ExtendedBetaBinomial`, and distributions
    returned by :func:`infection_dist`.

    This is useful for sampling from very large ``total_count``.

    This is used internally by
    :class:`~pyro.contrib.epidemiology.compartmental.CompartmentalModel`.

    :param thresh: New temporary threshold.
    :type thresh: int or float.
    """
    assert isinstance(thresh, (float, int))
    assert thresh > 0
    old = dist.Binomial.approx_sample_thresh
    try:
        dist.Binomial.approx_sample_thresh = thresh
        yield
    finally:
        dist.Binomial.approx_sample_thresh = old


@contextmanager
def set_approx_log_prob_tol(tol):
    """
    EXPERIMENTAL Context manager / decorator to temporarily set the global
    default value of ``Binomial.approx_log_prob_tol`` and
    ``BetaBinomial.approx_log_prob_tol``, thereby decreasing the computational
    complexity of scoring :class:`~pyro.distributions.Binomial` and
    :class:`~pyro.distributions.BetaBinomial` distributions.

    This is used internally by
    :class:`~pyro.contrib.epidemiology.compartmental.CompartmentalModel`.

    :param tol: New temporary tolold.
    :type tol: int or float.
    """
    assert isinstance(tol, (float, int))
    assert tol >= 0
    old1 = dist.Binomial.approx_log_prob_tol
    old2 = dist.BetaBinomial.approx_log_prob_tol
    try:
        dist.Binomial.approx_log_prob_tol = tol
        dist.BetaBinomial.approx_log_prob_tol = tol
        yield
    finally:
        dist.Binomial.approx_log_prob_tol = old1
        dist.BetaBinomial.approx_log_prob_tol = old2


@contextmanager
def set_relaxed_distributions(relaxed=True):
    global _RELAX
    old = _RELAX
    try:
        _RELAX = relaxed
        yield
    finally:
        _RELAX = old


def _validate_overdispersion(overdispersion):
    if is_validation_enabled():
        if not _all(0 <= overdispersion):
            raise ValueError("Expected overdispersion >= 0")
        if not _all(overdispersion < 2):
            raise ValueError("Expected overdispersion < 2")


def _relaxed_binomial(total_count, probs):
    """
    Returns a moment-matched :class:`~pyro.distributions.Normal` approximating
    a :class:`~pyro.distributions.Binomial` but allowing arbitrary real
    ``total_count`` and lower-bounding variance.
    """
    total_count, probs = broadcast_all(total_count, probs)

    mean = probs * total_count
    variance = total_count * probs * (1 - probs)

    scale = variance.clamp(min=_RELAX_MIN_VARIANCE).sqrt()
    return dist.Normal(mean, scale)


def _relaxed_beta_binomial(concentration1, concentration0, total_count):
    """
    Returns a moment-matched :class:`~pyro.distributions.Normal` approximating
    a :class:`~pyro.distributions.BetaBinomial` but allowing arbitrary real
    ``total_count`` and lower-bounding variance.
    """
    concentration1, concentration0, total_count = broadcast_all(
        concentration1, concentration0, total_count)

    c = concentration1 + concentration0
    beta_mean = concentration1 / c
    beta_variance = concentration1 * concentration0 / (c * c * (c + 1))
    mean = beta_mean * total_count
    variance = beta_variance * total_count * (c + total_count)

    scale = variance.clamp(min=_RELAX_MIN_VARIANCE).sqrt()
    return dist.Normal(mean, scale)


def binomial_dist(total_count, probs, *,
                  overdispersion=0.):
    """
    Returns a Beta-Binomial distribution that is an overdispersed version of a
    Binomial distribution, according to a parameter ``overdispersion``,
    typically set in the range 0.1 to 0.5.

    This is useful for (1) fitting real data that is overdispersed relative to
    a Binomial distribution, and (2) relaxing models of large populations to
    improve inference. In particular the ``overdispersion`` parameter lower
    bounds the relative uncertainty in stochastic models such that increasing
    population leads to a limiting scale-free dynamical system with bounded
    stochasticity, in contrast to Binomial-based SDEs that converge to
    deterministic ODEs in the large population limit.

    This parameterization satisfies the following properties:

    1.  Variance increases monotonically in ``overdispersion``.
    2.  ``overdispersion = 0`` results in a Binomial distribution.
    3.  ``overdispersion`` lower bounds the relative uncertainty ``std_dev /
        (total_count * p * q)``, where ``probs = p = 1 - q``, and serves as an
        asymptote for relative uncertainty as ``total_count → ∞``. This
        contrasts the Binomial whose relative uncertainty tends to zero.
    4.  If ``X ~ binomial_dist(n, p, overdispersion=σ)`` then in the large
        population limit ``n → ∞``, the scaled random variable ``X / n``
        converges in distribution to ``LogitNormal(log(p/(1-p)), σ)``.

    To achieve these properties we set ``p = probs``, ``q = 1 - p``, and::

        concentration = 1 / (p * q * overdispersion**2) - 1

    :param total_count: Number of Bernoulli trials.
    :type total_count: int or torch.Tensor
    :param probs: Event probabilities.
    :type probs: float or torch.Tensor
    :param overdispersion: Amount of overdispersion, in the half open interval
        [0,2). Defaults to zero.
    :type overdispersion: float or torch.tensor
    """
    _validate_overdispersion(overdispersion)
    if _is_zero(overdispersion):
        if _RELAX:
            return _relaxed_binomial(total_count, probs)
        return dist.ExtendedBinomial(total_count, probs)

    p = probs
    q = 1 - p
    od2 = (overdispersion + 1e-8) ** 2
    concentration1 = 1 / (q * od2 + 1e-8) - p
    concentration0 = 1 / (p * od2 + 1e-8) - q
    # At this point we have
    #   concentration1 + concentration0 == 1 / (p + q + od2 + 1e-8) - 1

    if _RELAX:
        return _relaxed_beta_binomial(concentration1, concentration0, total_count)
    return dist.ExtendedBetaBinomial(concentration1, concentration0, total_count)


def beta_binomial_dist(concentration1, concentration0, total_count, *,
                       overdispersion=0.):
    """
    Returns a Beta-Binomial distribution that is an overdispersed version of a
    the usual Beta-Binomial distribution, according to an extra parameter
    ``overdispersion``, typically set in the range 0.1 to 0.5.

    :param concentration1: 1st concentration parameter (alpha) for the
        Beta distribution.
    :type concentration1: float or torch.Tensor
    :param concentration0: 2nd concentration parameter (beta) for the
        Beta distribution.
    :type concentration0: float or torch.Tensor
    :param total_count: Number of Bernoulli trials.
    :type total_count: float or torch.Tensor
    :param overdispersion: Amount of overdispersion, in the half open interval
        [0,2). Defaults to zero.
    :type overdispersion: float or torch.tensor
    """
    _validate_overdispersion(overdispersion)
    if not _is_zero(overdispersion):
        # Compute harmonic sum of two sources of concentration resulting in
        # final concentration c = 1 / (1 / c_1 + 1 / c_2)
        od2 = (overdispersion + 1e-8) ** 2
        c_1 = concentration1 + concentration0
        c_2 = c_1 ** 2 / (concentration1 * concentration0 * od2 + 1e-8) - 1
        factor = 1 + c_1 / c_2
        concentration1 = concentration1 / factor
        concentration0 = concentration0 / factor

    if _RELAX:
        return _relaxed_beta_binomial(concentration1, concentration0, total_count)
    return dist.ExtendedBetaBinomial(concentration1, concentration0, total_count)


def poisson_dist(rate, *, overdispersion=0.):
    _validate_overdispersion(overdispersion)
    if _is_zero(overdispersion):
        return dist.Poisson(rate)
    raise NotImplementedError("TODO return a NegativeBinomial or GammaPoisson")


def negative_binomial_dist(concentration, probs=None, *,
                           logits=None, overdispersion=0.):
    _validate_overdispersion(overdispersion)
    if _is_zero(overdispersion):
        return dist.NegativeBinomial(concentration, probs=probs, logits=logits)
    raise NotImplementedError("TODO return a NegativeBinomial or GammaPoisson")


def infection_dist(*,
                   individual_rate,
                   num_infectious,
                   num_susceptible=math.inf,
                   population=math.inf,
                   concentration=math.inf,
                   overdispersion=0.):
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
    :param overdispersion: Amount of overdispersion, in the half open interval
        [0,2). Defaults to zero.
    :type overdispersion: float or torch.tensor
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
            return poisson_dist(R * I, overdispersion=overdispersion)
        else:
            # Return an overdispersed Negative-Binomial distribution.
            combined_k = k * I
            logits = torch.as_tensor(R / k).log()
            return negative_binomial_dist(combined_k, logits=logits,
                                          overdispersion=overdispersion)
    else:
        # Compute the probability that any given (susceptible, infectious)
        # pair of individuals results in an infection at this time step.
        p = torch.as_tensor(R / N).clamp(max=1 - 1e-6)
        # Combine infections from all individuals.
        combined_p = p.neg().log1p().mul(I).expm1().neg()  # = 1 - (1 - p)**I
        combined_p = combined_p.clamp(min=1e-6)

        if isinstance(k, float) and k == math.inf:
            # Return a pure Binomial model, combining the independent Binomial
            # models of each infectious individual.
            return binomial_dist(S, combined_p, overdispersion=overdispersion)
        else:
            # Return an overdispersed Beta-Binomial model, combining
            # independent BetaBinomial(c1,c0,S) models for each infectious
            # individual.
            c1 = (k * I).clamp(min=1e-6)
            c0 = c1 * (combined_p.reciprocal() - 1).clamp(min=1e-6)
            return beta_binomial_dist(c1, c0, S, overdispersion=overdispersion)
