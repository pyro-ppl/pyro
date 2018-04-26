from __future__ import absolute_import, division, print_function

import argparse
import logging
import math

import numpy as np
import pandas as pd
import torch

import pyro
import pyro.poutine as poutine
from pyro.distributions import Binomial, HalfCauchy, Normal, Uniform
from pyro.distributions.util import log_sum_exp
from pyro.infer import EmpiricalMarginal
from pyro.infer.abstract_infer import TracePredictive
from pyro.infer.mcmc import MCMC, NUTS

"""
Example has been adapted from [1]. It demonstrates how to do Bayesian inference using
NUTS (or, HMC) in Pyro, and use of some common inference utilities.

As in the Stan tutorial, this uses the small baseball dataset of Efron and Morris [2]
to estimate players' batting average which is the fraction of times a player got a
base hit out of the number of times they went up at bat.

The dataset separates the initial 45 at-bats statistics from the remaining season.
We use the hits data from the initial 45 at-bats to estimate the batting average
for each player. We then use the remaining season's data to validate the predictions
from our models.

Three models are evaluated:
 - Complete pooling model: The success probability of scoring a hit is shared
     amongst all players.
 - No pooling model: Each individual player's success probability is distinct and
     there is no data sharing amongst players.
 - Partial pooling model: A hierarchical model with partial data sharing.


We recommend Radford Neal's tutorial on HMC ([3]) to users who would like to get a
more comprehensive understanding of HMC and its variants, and to [4] for details on
the No U-Turn Sampler, which provides an efficient and automated way (i.e. limited
hyper-parameters) of running HMC on different problems.

[1] Carpenter B. (2016), ["Hierarchical Partial Pooling for Repeated Binary Trials"]
    (http://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html).
[2] Efron B., Morris C. (1975), "Data analysis using Stein's estimator and its
    generalizations", J. Amer. Statist. Assoc., 70, 311-319.
[3] Neal, R. (2012), "MCMC using Hamiltonian Dynamics",
    (https://arxiv.org/pdf/1206.1901.pdf)
[4] Hoffman, M. D. and Gelman, A. (2014), "The No-U-turn sampler: Adaptively setting
    path lengths in Hamiltonian Monte Carlo", (https://arxiv.org/abs/1111.4246)
"""

logging.basicConfig(format='%(message)s', level=logging.INFO)
# Enable validation checks
pyro.enable_validation(True)
pyro.set_rng_seed(1)
DATA_URL = "https://d2fefpcigoriu7.cloudfront.net/datasets/EfronMorrisBB.txt"


# ===================================
#               MODELS
# ===================================


def fully_pooled(at_bats):
    """
    Number of hits in $K$ at bats for each player has a Binomial
    distribution with a common probability of success, $\phi$.

    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :return: Number of hits predicted by the model.
    """
    phi_prior = Uniform(at_bats.new_tensor(0), at_bats.new_tensor(1))
    phi = pyro.sample("phi", phi_prior)
    return pyro.sample("obs", Binomial(at_bats, phi))


def not_pooled(at_bats):
    """
    Number of hits in $K$ at bats for each player has a Binomial
    distribution with independent probability of success, $\phi_i$.

    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :return: Number of hits predicted by the model.
    """
    num_players = at_bats.shape[0]
    phi_prior = Uniform(at_bats.new_tensor(0), at_bats.new_tensor(1)).expand_by([num_players]).independent(1)
    phi = pyro.sample("phi", phi_prior)
    return pyro.sample("obs", Binomial(at_bats, phi))


def partially_pooled(at_bats):
    """
    Number of hits has a Binomial distribution with a logit link function.
    The logits $\alpha$ for each player is normally distributed with the
    mean and scale parameters sharing a common prior.

    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :return: Number of hits predicted by the model.
    """
    num_players = at_bats.shape[0]
    loc = pyro.sample("loc", Normal(at_bats.new_tensor(-1), at_bats.new_tensor(1)))
    scale = pyro.sample("scale", HalfCauchy(at_bats.new_tensor(0), at_bats.new_tensor(1)))
    alpha = pyro.sample("alpha", Normal(loc, scale).expand_by([num_players]).independent(1))
    return pyro.sample("obs", Binomial(at_bats, logits=alpha))


def conditioned_model(model, at_bats, hits):
    """
    Condition the model on observed data, for inference.

    :param model: python callable with Pyro primitives.
    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :param (torch.Tensor) hits: Number of hits for the given at bats.
    """
    return poutine.condition(model, data={"obs": hits})(at_bats)


# ===================================
#        DATA SUMMARIZE UTILS
# ===================================


def get_site_stats(array, player_names):
    """
    Return the summarized statistics for a given array corresponding
    to the values sampled for a latent or response site.
    """
    if len(array.shape) == 1:
        df = pd.DataFrame(array).transpose()
    else:
        df = pd.DataFrame(array, columns=player_names).transpose()
    return df.apply(pd.Series.describe, axis=1)[["mean", "std", "25%", "50%", "75%"]]


def summary(traces, sites, player_names, transforms={}):
    """
    Return summarized statistics for each of the ``sites`` in the
    traces corresponding to the approximate posterior.
    """
    marginal = EmpiricalMarginal(traces, sites).get_samples_and_weights()[0].numpy()
    site_stats = {}
    for i in range(marginal.shape[1]):
        site_name = sites[i]
        marginal_site = marginal[:, i]
        if site_name in transforms:
            marginal_site = transforms[site_name](marginal_site)
        site_stats[site_name] = get_site_stats(marginal_site, player_names)
    return site_stats


def train_test_split(pd_dataframe):
    """
    Training data - 45 initial at-bats and hits for each player.
    Validation data - Full season at-bats and hits for each player.
    """
    train_data = torch.tensor(pd_dataframe.as_matrix(["At-Bats", "Hits"]), dtype=torch.float)
    test_data = torch.tensor(pd_dataframe.as_matrix(["SeasonAt-Bats", "SeasonHits"]), dtype=torch.float)
    first_name = pd_dataframe["FirstName"].values
    last_name = pd_dataframe["LastName"].values
    player_names = [" ".join([first, last]) for first, last in zip(first_name, last_name)]
    return train_data, test_data, player_names


# ===================================
#       MODEL EVALUATION UTILS
# ===================================


def sample_posterior_predictive(posterior_predictive, baseball_dataset):
    """
    Generate samples from posterior predictive distribution.
    """
    train, test, player_names = train_test_split(baseball_dataset)
    at_bats = train[:, 0]
    at_bats_season = test[:, 0]
    logging.Formatter("%(message)s")
    logging.info("\nPosterior Predictive:")
    logging.info("Hit Rate - Initial 45 At Bats")
    logging.info("-----------------------------")
    train_predict = posterior_predictive.run(at_bats)
    train_summary = summary(train_predict, sites=["obs"], player_names=player_names)["obs"]
    train_summary = train_summary.assign(ActualHits=baseball_dataset[["Hits"]].values)
    logging.info(train_summary)
    logging.info("\nHit Rate - Season Predictions")
    logging.info("-----------------------------")
    test_predict = posterior_predictive.run(at_bats_season)
    test_summary = summary(test_predict, sites=["obs"], player_names=player_names)["obs"]
    test_summary = test_summary.assign(ActualHits=baseball_dataset[["SeasonHits"]].values)
    logging.info(test_summary)


def evaluate_log_predictive_density(model, model_trace_posterior, baseball_dataset):
    """
    Evaluate the log probability density of observing the unseen data (season hits)
    given a model and empirical distribution over the parameters.
    """
    _, test, player_names = train_test_split(baseball_dataset)
    at_bats_season, hits_season = test[:, 0], test[:, 1]
    test_eval = TracePredictive(conditioned_model,
                                model_trace_posterior,
                                num_samples=args.num_samples)
    test_eval.run(model, at_bats_season, hits_season)
    trace_log_pdf = []
    for tr in test_eval.exec_traces:
        trace_log_pdf.append(tr.log_prob_sum())
    # Use LogSumExp trick to evaluate $log(1/num_samples \sum_i p(new_data | \theta^{i})) $,
    # where $\theta^{i}$ are parameter samples from the model's posterior.
    posterior_pred_density = log_sum_exp(torch.stack(trace_log_pdf)) - math.log(len(trace_log_pdf))
    logging.info("\nLog posterior predictive density")
    logging.info("---------------------------------")
    logging.info("{:.4f}\n".format(posterior_pred_density))


def main(args):
    baseball_dataset = pd.read_csv(DATA_URL, "\t")
    train, _, player_names = train_test_split(baseball_dataset)
    at_bats, hits = train[:, 0], train[:, 1]
    nuts_kernel = NUTS(conditioned_model, adapt_step_size=True)
    logging.info("Original Dataset:")
    logging.info(baseball_dataset)

    # (1) Full Pooling Model
    posterior_fully_pooled = MCMC(nuts_kernel, num_samples=args.num_samples, warmup_steps=args.warmup_steps) \
        .run(fully_pooled, at_bats, hits)
    logging.info("\nModel: Fully Pooled")
    logging.info("===================")
    logging.info("\nphi:")
    logging.info(summary(posterior_fully_pooled, sites=["phi"], player_names=player_names)["phi"])
    posterior_predictive = TracePredictive(fully_pooled,
                                           posterior_fully_pooled,
                                           num_samples=args.num_samples)
    sample_posterior_predictive(posterior_predictive, baseball_dataset)
    evaluate_log_predictive_density(fully_pooled, posterior_fully_pooled, baseball_dataset)

    # (2) No Pooling Model
    posterior_not_pooled = MCMC(nuts_kernel, num_samples=args.num_samples, warmup_steps=args.warmup_steps) \
        .run(not_pooled, at_bats, hits)
    logging.info("\nModel: Not Pooled")
    logging.info("=================")
    logging.info("\nphi:")
    logging.info(summary(posterior_not_pooled, sites=["phi"], player_names=player_names)["phi"])
    posterior_predictive = TracePredictive(not_pooled,
                                           posterior_not_pooled,
                                           num_samples=args.num_samples)
    sample_posterior_predictive(posterior_predictive, baseball_dataset)
    evaluate_log_predictive_density(not_pooled, posterior_not_pooled, baseball_dataset)

    # (3) Partially Pooled Model
    posterior_partially_pooled = MCMC(nuts_kernel, num_samples=args.num_samples, warmup_steps=args.warmup_steps) \
        .run(partially_pooled, at_bats, hits)
    logging.info("\nModel: Partially Pooled")
    logging.info("=======================")
    logging.info("\nSigmoid(alpha):")
    logging.info(summary(posterior_partially_pooled,
                         sites=["alpha"],
                         player_names=player_names,
                         transforms={"alpha": lambda x: 1. / (1 + np.exp(-x))})["alpha"])
    posterior_predictive = TracePredictive(partially_pooled,
                                           posterior_partially_pooled,
                                           num_samples=args.num_samples)
    sample_posterior_predictive(posterior_predictive, baseball_dataset)
    evaluate_log_predictive_density(partially_pooled, posterior_partially_pooled, baseball_dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseball batting average using HMC")
    parser.add_argument("-n", "--num-samples", nargs="?", default=1200, type=int)
    parser.add_argument("--warmup-steps", nargs='?', default=300, type=int)
    parser.add_argument("--rng_seed", nargs='?', default=0, type=int)
    args = parser.parse_args()
    main(args)
