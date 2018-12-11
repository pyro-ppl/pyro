from __future__ import absolute_import, division, print_function

import argparse
import logging
import math
import os

import pandas as pd
import torch

import pyro
from pyro.distributions import Beta, Binomial, HalfCauchy, Normal, Pareto, Uniform
from pyro.distributions.util import logsumexp
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

# work around with the error "RuntimeError: received 0 items of ancdata"
# see https://discuss.pytorch.org/t/received-0-items-of-ancdata-pytorch-0-4-0/19823
torch.multiprocessing.set_sharing_strategy('file_system')
logging.basicConfig(format='%(message)s', level=logging.INFO)
# Enable validation checks
pyro.enable_validation(True)
DATA_URL = "https://d2fefpcigoriu7.cloudfront.net/datasets/EfronMorrisBB.txt"


# ===================================
#               MODELS
# ===================================


def fully_pooled(at_bats, hits):
    r"""
    Number of hits in $K$ at bats for each player has a Binomial
    distribution with a common probability of success, $\phi$.

    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :param (torch.Tensor) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    phi_prior = Uniform(at_bats.new_tensor(0), at_bats.new_tensor(1))
    phi = pyro.sample("phi", phi_prior)
    return pyro.sample("obs", Binomial(at_bats, phi), obs=hits)


def not_pooled(at_bats, hits):
    r"""
    Number of hits in $K$ at bats for each player has a Binomial
    distribution with independent probability of success, $\phi_i$.

    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :param (torch.Tensor) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    num_players = at_bats.shape[0]
    with pyro.plate("num_players", num_players):
        phi_prior = Uniform(at_bats.new_tensor(0), at_bats.new_tensor(1))
        phi = pyro.sample("phi", phi_prior)
        return pyro.sample("obs", Binomial(at_bats, phi), obs=hits)


def partially_pooled(at_bats, hits):
    r"""
    Number of hits has a Binomial distribution with independent
    probability of success, $\phi_i$. Each $\phi_i$ follows a Beta
    distribution with concentration parameters $c_1$ and $c_2$, where
    $c_1 = m * kappa$, $c_2 = (1 - m) * kappa$, $m ~ Uniform(0, 1)$,
    and $kappa ~ Pareto(1, 1.5)$.

    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :param (torch.Tensor) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    num_players = at_bats.shape[0]
    m = pyro.sample("m", Uniform(at_bats.new_tensor(0), at_bats.new_tensor(1)))
    kappa = pyro.sample("kappa", Pareto(at_bats.new_tensor(1), at_bats.new_tensor(1.5)))
    with pyro.plate("num_players", num_players):
        phi_prior = Beta(m * kappa, (1 - m) * kappa)
        phi = pyro.sample("phi", phi_prior)
        return pyro.sample("obs", Binomial(at_bats, phi), obs=hits)


def partially_pooled_with_logit(at_bats, hits):
    r"""
    Number of hits has a Binomial distribution with a logit link function.
    The logits $\alpha$ for each player is normally distributed with the
    mean and scale parameters sharing a common prior.

    :param (torch.Tensor) at_bats: Number of at bats for each player.
    :param (torch.Tensor) hits: Number of hits for the given at bats.
    :return: Number of hits predicted by the model.
    """
    num_players = at_bats.shape[0]
    loc = pyro.sample("loc", Normal(at_bats.new_tensor(-1), at_bats.new_tensor(1)))
    scale = pyro.sample("scale", HalfCauchy(scale=at_bats.new_tensor(1)))
    with pyro.plate("num_players", num_players):
        alpha = pyro.sample("alpha", Normal(loc, scale))
        return pyro.sample("obs", Binomial(at_bats, logits=alpha), obs=hits)


# ===================================
#        DATA SUMMARIZE UTILS
# ===================================


def get_site_stats(array, player_names):
    """
    Return the summarized statistics for a given array corresponding
    to the values sampled for a latent or response site.
    """
    # TODO: only use pandas (or any lightweight tabular package) to display final result
    if len(array.shape) == 1:
        df = pd.DataFrame(array).transpose()
    else:
        df = pd.DataFrame(array, columns=player_names).transpose()
    return df.apply(pd.Series.describe, axis=1)[["mean", "std", "25%", "50%", "75%"]]


def summary(trace_posterior, sites, player_names, transforms={}, diagnostics=True):
    """
    Return summarized statistics for each of the ``sites`` in the
    traces corresponding to the approximate posterior.
    """
    marginal = trace_posterior.marginal(sites)
    site_stats = {}
    for site_name in sites:
        marginal_site = marginal.support(flatten=True)[site_name]
        if site_name in transforms:
            marginal_site = transforms[site_name](marginal_site)

        site_stats[site_name] = get_site_stats(marginal_site.numpy(), player_names)
        if diagnostics and trace_posterior.num_chains > 1:
            diag = marginal.diagnostics()[site_name]
            site_stats[site_name] = site_stats[site_name].assign(n_eff=diag["n_eff"].numpy(),
                                                                 r_hat=diag["r_hat"].numpy())
    return site_stats


def train_test_split(pd_dataframe):
    """
    Training data - 45 initial at-bats and hits for each player.
    Validation data - Full season at-bats and hits for each player.
    """
    train_data = torch.tensor(pd_dataframe[["At-Bats", "Hits"]].values, dtype=torch.float)
    test_data = torch.tensor(pd_dataframe[["SeasonAt-Bats", "SeasonHits"]].values, dtype=torch.float)
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
    # set hits=None to convert it from observation node to sample node
    train_predict = posterior_predictive.run(at_bats, None)
    train_summary = summary(train_predict, sites=["obs"],
                            player_names=player_names, diagnostics=False)["obs"]
    train_summary = train_summary.assign(ActualHits=baseball_dataset[["Hits"]].values)
    logging.info(train_summary)
    logging.info("\nHit Rate - Season Predictions")
    logging.info("-----------------------------")
    test_predict = posterior_predictive.run(at_bats_season, None)
    test_summary = summary(test_predict, sites=["obs"],
                           player_names=player_names, diagnostics=False)["obs"]
    test_summary = test_summary.assign(ActualHits=baseball_dataset[["SeasonHits"]].values)
    logging.info(test_summary)


def evaluate_log_predictive_density(posterior_predictive, baseball_dataset):
    """
    Evaluate the log probability density of observing the unseen data (season hits)
    given a model and empirical distribution over the parameters.
    """
    _, test, player_names = train_test_split(baseball_dataset)
    at_bats_season, hits_season = test[:, 0], test[:, 1]
    test_eval = posterior_predictive.run(at_bats_season, hits_season)
    trace_log_pdf = []
    for tr in test_eval.exec_traces:
        trace_log_pdf.append(tr.log_prob_sum())
    # Use LogSumExp trick to evaluate $log(1/num_samples \sum_i p(new_data | \theta^{i})) $,
    # where $\theta^{i}$ are parameter samples from the model's posterior.
    posterior_pred_density = logsumexp(torch.stack(trace_log_pdf), dim=-1) - math.log(len(trace_log_pdf))
    logging.info("\nLog posterior predictive density")
    logging.info("--------------------------------")
    logging.info("{:.4f}\n".format(posterior_pred_density))


def main(args):
    pyro.set_rng_seed(args.rng_seed)
    baseball_dataset = pd.read_csv(DATA_URL, "\t")
    train, _, player_names = train_test_split(baseball_dataset)
    at_bats, hits = train[:, 0], train[:, 1]
    logging.info("Original Dataset:")
    logging.info(baseball_dataset)
    num_predictive_samples = args.num_samples * args.num_chains

    # (1) Full Pooling Model
    nuts_kernel = NUTS(fully_pooled)
    posterior_fully_pooled = MCMC(nuts_kernel,
                                  num_samples=args.num_samples,
                                  warmup_steps=args.warmup_steps,
                                  num_chains=args.num_chains).run(at_bats, hits)
    logging.info("\nModel: Fully Pooled")
    logging.info("===================")
    logging.info("\nphi:")
    logging.info(summary(posterior_fully_pooled, sites=["phi"], player_names=player_names)["phi"])
    posterior_predictive = TracePredictive(fully_pooled,
                                           posterior_fully_pooled,
                                           num_samples=num_predictive_samples)
    sample_posterior_predictive(posterior_predictive, baseball_dataset)
    evaluate_log_predictive_density(posterior_predictive, baseball_dataset)

    # (2) No Pooling Model
    nuts_kernel = NUTS(not_pooled)
    posterior_not_pooled = MCMC(nuts_kernel,
                                num_samples=args.num_samples,
                                warmup_steps=args.warmup_steps,
                                num_chains=args.num_chains).run(at_bats, hits)
    logging.info("\nModel: Not Pooled")
    logging.info("=================")
    logging.info("\nphi:")
    logging.info(summary(posterior_not_pooled, sites=["phi"], player_names=player_names)["phi"])
    posterior_predictive = TracePredictive(not_pooled,
                                           posterior_not_pooled,
                                           num_samples=num_predictive_samples)
    sample_posterior_predictive(posterior_predictive, baseball_dataset)
    evaluate_log_predictive_density(posterior_predictive, baseball_dataset)

    # (3) Partially Pooled Model
    # TODO: remove once htps://github.com/uber/pyro/issues/1458 is resolved
    if "CI" not in os.environ:
        nuts_kernel = NUTS(partially_pooled)
        posterior_partially_pooled = MCMC(nuts_kernel,
                                          num_samples=args.num_samples,
                                          warmup_steps=args.warmup_steps,
                                          num_chains=args.num_chains).run(at_bats, hits)
        logging.info("\nModel: Partially Pooled")
        logging.info("=======================")
        logging.info("\nphi:")
        logging.info(summary(posterior_partially_pooled, sites=["phi"],
                             player_names=player_names)["phi"])
        posterior_predictive = TracePredictive(partially_pooled,
                                               posterior_partially_pooled,
                                               num_samples=num_predictive_samples)
        sample_posterior_predictive(posterior_predictive, baseball_dataset)
        evaluate_log_predictive_density(posterior_predictive, baseball_dataset)

    # (4) Partially Pooled with Logit Model
    nuts_kernel = NUTS(partially_pooled_with_logit)
    posterior_partially_pooled_with_logit = MCMC(nuts_kernel,
                                                 num_samples=args.num_samples,
                                                 warmup_steps=args.warmup_steps,
                                                 num_chains=args.num_chains).run(at_bats, hits)
    logging.info("\nModel: Partially Pooled with Logit")
    logging.info("==================================")
    logging.info("\nSigmoid(alpha):")
    logging.info(summary(posterior_partially_pooled_with_logit,
                         sites=["alpha"],
                         player_names=player_names,
                         transforms={"alpha": lambda x: 1. / (1 + (-x).exp())})["alpha"])
    posterior_predictive = TracePredictive(partially_pooled_with_logit,
                                           posterior_partially_pooled_with_logit,
                                           num_samples=num_predictive_samples)
    sample_posterior_predictive(posterior_predictive, baseball_dataset)
    evaluate_log_predictive_density(posterior_predictive, baseball_dataset)


if __name__ == "__main__":
    assert pyro.__version__.startswith('0.3.0')
    parser = argparse.ArgumentParser(description="Baseball batting average using HMC")
    parser.add_argument("-n", "--num-samples", nargs="?", default=200, type=int)
    parser.add_argument("--num-chains", nargs='?', default=4, type=int)
    parser.add_argument("--warmup-steps", nargs='?', default=100, type=int)
    parser.add_argument("--rng_seed", nargs='?', default=0, type=int)
    parser.add_argument('--jit', action='store_true', default=False,
                        help='use PyTorch jit')
    args = parser.parse_args()
    main(args)
