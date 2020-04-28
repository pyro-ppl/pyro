# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import pyro.distributions as dist


def infection_dist(*,
                   individual_rate,
                   num_infectious,
                   num_susceptible=math.inf,
                   population=math.inf,
                   concentration=math.inf):
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
    :concentration: The concentration or dispersion parameter ``k`` in
        overdispersed models of superspreaders [1,2]. This defaults to minimum
        variance ``concentration = ∞``.
    """
    # Convert to colloquial variable names.
    R = individual_rate
    I = num_infectious
    S = num_susceptible
    N = population
    k = concentration

    if population == math.inf:
        if k == math.inf:
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

        if k == math.inf:
            # Return a pure Binomial model, combining the independent Binomial
            # models of each infectious individual.
            return dist.ExtendedBinomial(S, combined_p)
        else:
            # Return an overdispersed Beta-Binomial model, combining
            # independent BetaBinomial(c1,c0,S) models for each infectious
            # individual.
            c1 = k * I
            c0 = c1 * (combined_p.reciprocal() - 1).clamp(min=1e-6)
            return dist.ExtendedBetaBinomial(c1, c0, S)
