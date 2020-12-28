# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
from timeit import default_timer

import torch
import torch.nn as nn
from opt_einsum import contract as einsum
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.infer.autoguide import AutoLowRankMultivariateNormal, AutoNormal
from pyro.infer.reparam import HaarReparam
from pyro.optim import ClippedAdam

logger = logging.getLogger(__name__)


def MomentMatchedPoisson(rate):
    # Poisson mean = variance = rate
    # LogNormal(m,s)
    #   mean = exp(m + s**2/2)
    #   variance = (exp(s**2) - 1) exp(2*m + s**2)
    # Solving for (m,s) given rate:
    #   m + s**2/2 = log(rate) = 2*m + s**2 + log(exp(s*2) - 1)
    # ==> log(rate) = -log(exp(s**2) - 1)
    # ==> 1/rate = exp(s**2) - 1
    # ==> s = sqrt(log(1 + 1/rate))
    # ==> m = log(rate) - s**2/2
    s2 = rate.reciprocal().log1p()
    m = rate.log() - s2 / 2
    s = s2.sqrt()
    return dist.LogNormal(m, s)


class TimeSpaceStrainModel(nn.Module):
    r"""
    Phylogeographic model to track strains over space and time.

    This model fuses three types of data:

    1.  **Aggregate epidemiological data** in the form of case counts and death
        counts in each (time, region) bucket.
    2.  **Transit data** in the form of a number of covariates believed to
        predict inter-region infection rates. These might combine measured data
        such as flight information and mobile phone records with prior
        covariates such as whether a pair of regions share a border.
    3.  **Genetic sequence** data from infected individuals with known (time,
        region) identity. This model assumes genetic samples are collected
        uniformly within each region, but allows for arbitrary collection rates
        across regions; this model this avoids cross-region bias of sample
        collection.

    Inference has complexity ``O(T * R * S)``, where ``T`` is the number of
    time steps, ``R`` is the number of regions, and ``S`` is the number of
    strains. Inference complexity does not depend on the number of genetic
    samples forming leaves of the phylogeny; therefore this method is suitable
    for coarse phylogenies with ~1000 internal nodes but millions of leaves.
    To ensure the model fits in memory, you might choose coarse time steps of
    weeks for ~100 time steps, coarse regions with ~100-1000 regions (possibly
    with finer detail in a particular region of interest, e.g. foreign
    countries + local provinces), and coarse phylogenies with ~100-1000
    strains.

    :param Tensor case_data: A ``(T,R)``-shaped tensor of confirmed case counts
        in each (time,region) bucket.
    :param Tensor death_data: A ``(T,R)``-shaped tensor of confirmed death
        counts in each (time,region) bucket.
    :param Tensor transit_data: A ``(T,R,R,P)``-shaped tensor of ``P``-many
        covariates, each defining time-dependent region-to-region transition
        rates. Values must be nonnegative.
    :param Tensor sample_time:
    :param Tensor sample_region:
    :param Tensor sample_strain: Three integer vectors of shape ``(N,)``
        containing the time, region, and strain classification of each of ``N``
        genetic samples.
    :param Tensor strain_distance: An ``(S,S)``-shaped array of genetic
        distances between each pair of ``S``-many strains. This could be
        constructed e.g. by estimating a coarse phylogeny among strains and
        measuring the edge distance between each pair of strains.
    """
    def __init__(
        self,
        case_data,
        death_data,
        transit_data,
        sample_time,
        sample_region,
        sample_strain,
        strain_distance,
        death_rate,
    ):
        N = len(sample_time)
        assert sample_time.shape == (N,)
        assert sample_region.shape == (N,)
        assert sample_strain.shape == (N,)
        S = 1 + sample_strain.max().item()
        assert strain_distance.shape == (S, S)
        T, R = case_data.shape
        assert death_data.shape == (T, R)
        P = transit_data.size(-1)
        assert transit_data.shape == (T, R, R, P)
        assert isinstance(death_rate, float) and 0 < death_rate < 1
        assert transit_data.min() >= 0, "transit data must be nonnegative"

        # Convert sparse sample data to dense multinomial observations.
        strain_data = torch.zeros(T, R, S)
        i = sample_time.mul(R).add_(sample_region).mul_(S).add_(sample_strain)
        one = torch.ones(()).expand_as(i)
        strain_data.reshape(-1).scatter_add_(0, i, one)
        strain_total = strain_data.sum(-1)

        self.num_time_steps = T
        self.num_regions = R
        self.num_strains = S
        self.num_transit_covariates = P

        self.register_buffer("case_data", case_data)
        self.register_buffer("death_data", death_data)
        self.register_buffer("transit_data", transit_data)
        self.register_buffer("strain_data", strain_data)
        self.register_buffer("strain_total", strain_total)
        self.register_buffer("strain_distance", strain_distance)
        self.death_rate = death_rate

    def model(self):
        T = self.num_time_steps
        P = self.num_transit_covariates
        time_plate = pyro.plate("time", T, dim=-3)
        step_plate = pyro.plate("step", T - 1, dim=-3)
        region_plate = pyro.plate("region", T, dim=-2)
        strain_plate = pyro.plate("strain", T, dim=-1)

        # Sample case counting parameters, factored over time x region.
        with time_plate:
            case_rate_time = pyro.sample("case_rate_time", dist.Beta(1, 2))
        with region_plate:
            case_rate_region = pyro.sample("case_rate_region", dist.Beta(1, 2))
        case_rate = case_rate_time * case_rate_region

        # Sample local spreading dynamics.
        # TODO model spatial structure, say hierarchically.
        R0 = pyro.sample("R0", dist.LogNormal(0, 1))
        R_scale = pyro.sample("R_scale", dist.LogNormal(0, 1))
        R_drift_scale = pyro.sample("R_drift_scale", dist.LogNormal(-2, 2))
        with time_plate, region_plate:
            Rt = pyro.sample("Rt", dist.LogNormal(R0, R_scale))
        with step_plate, region_plate:
            pyro.sample("R_drift", dist.LogNormal(0, R_drift_scale),
                        obs=Rt[1:] / Rt[:-1])

        # Sample inter-region spreading dynamics.
        transit_rate = pyro.sample("transit_rate",
                                   dist.Exponential(1).expand(P).to_event(1))

        # Sample mutation dynamics.
        mutation_rate = pyro.sample("mutation_rate", dist.LogNormal(-5, 5))
        mutation_scale = pyro.sample("mutation_scale", dist.LogNormal(0, 1))
        with strain_plate:
            concentration = ((-mutation_rate) * self.strain_distance).exp()
            concentration = concentration * mutation_scale
            strain_rate = pyro.sample("strain_rate",
                                      dist.Dirichlet(concentration))

        # Sample infections as a factor graph.
        with time_plate, region_plate, strain_plate:
            infections = pyro.sample(
                "infections",
                dist.ImproperUniform(constraints.positive, (), ()),
            )
        with pyro.plate("dt", T - 1, dim=-3), region_plate, strain_plate:
            pred = einsum(
                "trs,tr,trRp,p,sS->tRS",
                infections[:-1],
                Rt[:-1],
                self.transit_data[:-1],
                transit_rate,
                strain_rate,
            )
            pyro.sample("infections_step", MomentMatchedPoisson(pred),
                        obs=infections[1:])

        # The remainder of the model concerns time-region local observations.
        infections_sum = infections.sum(-1, True)
        strain_probs = infections / infections_sum
        with time_plate, region_plate:
            # Condition on case counts, marginalized over strains.
            # TODO use overdispersed distribution.
            pyro.sample("case_obs",
                        dist.Poission(infections_sum * case_rate),
                        obs=self.case_data)

            # Condition on death counts, marginalized over strains.
            # TODO use overdispersed distribution.
            pyro.sample("death_obs",
                        dist.Poission(infections_sum * self.death_rate),
                        obs=self.death_data)

            # Condition on strain counts.
            pyro.sample("strains",
                        dist.Multinomial(self.strain_total, strain_probs),
                        obs=self.strain_obs)

    def fit(
        self,
        *,
        haar=True,
        guide_rank=0,
        init_scale=0.01,
        learning_rate=0.02,
        learning_rate_decay=0.1,
        num_steps=1001,
        jit=False,
        log_every=100,
    ):
        """
        Fits a guide via stochastic variational inference.

        After this is called, the ``.guide`` attribute can
        be used to generate samples, medians, or quantiles.
        """
        # Configure variational inference.
        logger.info("Running inference...")
        model = self.model
        if haar:
            def time_reparam(site):
                if not site["is_observed"]:
                    return HaarReparam(dim=-3 - site["fn"].event_dim)
            model = poutine.reparam(model, time_reparam)
        if guide_rank == 0:
            guide = AutoNormal(model, init_scale=init_scale)
        elif guide_rank is None or isinstance(guide_rank, int):
            guide = AutoLowRankMultivariateNormal(model, init_scale=init_scale,
                                                  rank=guide_rank)
        else:
            raise ValueError(f"Invalid guide_rank: {guide_rank}")
        Elbo = JitTrace_ELBO if jit else Trace_ELBO
        elbo = Elbo(max_plate_nesting=3, ignore_jit_warnings=True)
        optim = ClippedAdam({"lr": learning_rate,
                             "lrd": learning_rate_decay ** (1 / num_steps)})
        svi = SVI(model, guide, optim, elbo)

        # Run inference.
        start_time = default_timer()
        losses = []
        for step in range(num_steps):
            loss = svi.step() / self.case_data.numel()
            losses.append(loss)
            if log_every and step % log_every == 100:
                logger.info("step {step: >5d} loss = {loss:0.4g}")
        elapsed = default_timer() - start_time
        logger.info("SVI took {:0.1f} seconds, {:0.1f} step/sec"
                    .format(elapsed, (1 + num_steps) / elapsed))

        self.guide = guide
        return losses
