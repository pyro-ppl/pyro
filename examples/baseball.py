import argparse
import logging
import math

import pandas as pd
import torch

import pyro
from pyro.distributions import Beta, Binomial, HalfCauchy, Normal, Pareto, Uniform
from pyro.distributions.util import scalar_like, sum_rightmost
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.util import predictive, initialize_model
from pyro.poutine.util import site_is_subsample
from pyro.util import ignore_experimental_warning

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
    phi_prior = Uniform(scalar_like(at_bats, 0), scalar_like(at_bats, 1))
    phi = pyro.sample("phi", phi_prior)
    num_players = at_bats.shape[0]
    with pyro.plate("num_players", num_players):
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
        phi_prior = Uniform(scalar_like(at_bats, 0), scalar_like(at_bats, 1))
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
    m = pyro.sample("m", Uniform(scalar_like(at_bats, 0), scalar_like(at_bats, 1)))
    kappa = pyro.sample("kappa", Pareto(scalar_like(at_bats, 1), scalar_like(at_bats, 1.5)))
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
    loc = pyro.sample("loc", Normal(scalar_like(at_bats, -1), scalar_like(at_bats, 1)))
    scale = pyro.sample("scale", HalfCauchy(scale=scalar_like(at_bats, 1)))
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


def summary(posterior, sites, player_names, transforms={}, diagnostics=None):
    """
    Return summarized statistics for each of the ``sites`` in the
    traces corresponding to the approximate posterior.
    """
    site_stats, diag_stats = {}, diagnostics

    for site_name in sites:
        marginal_site = posterior[site_name]

        if site_name in transforms:
            marginal_site = transforms[site_name](marginal_site)

        site_stats[site_name] = get_site_stats(marginal_site.cpu().numpy(), player_names)

        if diag_stats:
            site_stats[site_name] = site_stats[site_name].assign(
                n_eff=diag_stats[site_name]["n_eff"].cpu().numpy(),
                r_hat=diag_stats[site_name]["r_hat"].cpu().numpy())
    return site_stats


def train_test_split(pd_dataframe):
    """
    Training data - 45 initial at-bats and hits for each player.
    Validation data - Full season at-bats and hits for each player.
    """
    device = torch.Tensor().device
    train_data = torch.tensor(pd_dataframe[["At-Bats", "Hits"]].values, dtype=torch.float, device=device)
    test_data = torch.tensor(pd_dataframe[["SeasonAt-Bats", "SeasonHits"]].values, dtype=torch.float, device=device)
    first_name = pd_dataframe["FirstName"].values
    last_name = pd_dataframe["LastName"].values
    player_names = [" ".join([first, last]) for first, last in zip(first_name, last_name)]
    return train_data, test_data, player_names


# ===================================
#       MODEL EVALUATION UTILS
# ===================================


def sample_posterior_predictive(model, posterior_samples, baseball_dataset):
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
    with ignore_experimental_warning():
        train_predict = predictive(model, posterior_samples, at_bats, None)
    train_summary = summary(train_predict,
                            sites=["obs"],
                            player_names=player_names)["obs"]
    train_summary = train_summary.assign(ActualHits=baseball_dataset[["Hits"]].values)
    logging.info(train_summary)
    logging.info("\nHit Rate - Season Predictions")
    logging.info("-----------------------------")
    with ignore_experimental_warning():
        test_predict = predictive(model, posterior_samples, at_bats_season, None)
    test_summary = summary(test_predict,
                           sites=["obs"],
                           player_names=player_names)["obs"]
    test_summary = test_summary.assign(ActualHits=baseball_dataset[["SeasonHits"]].values)
    logging.info(test_summary)


def evaluate_log_posterior_density(model, posterior_samples, baseball_dataset):
    """
    Evaluate the log probability density of observing the unseen data (season hits)
    given a model and posterior distribution over the parameters.
    """
    _, test, player_names = train_test_split(baseball_dataset)
    at_bats_season, hits_season = test[:, 0], test[:, 1]
    with ignore_experimental_warning():
        trace = predictive(model, posterior_samples, at_bats_season, hits_season,
                           parallel=True, return_trace=True)
    # Use LogSumExp trick to evaluate $log(1/num_samples \sum_i p(new_data | \theta^{i})) $,
    # where $\theta^{i}$ are parameter samples from the model's posterior.
    trace.compute_log_prob()
    log_joint = 0.
    for name, site in trace.nodes.items():
        if site["type"] == "sample" and not site_is_subsample(site):
            # We use `sum_rightmost(x, -1)` to take the sum of all rightmost dimensions of `x`
            # except the first dimension (which corresponding to the number of posterior samples)
            site_log_prob_sum = sum_rightmost(site['log_prob'], -1)
            log_joint += site_log_prob_sum
    posterior_pred_density = torch.logsumexp(log_joint, dim=0) - math.log(log_joint.shape[0])
    logging.info("\nLog posterior predictive density")
    logging.info("--------------------------------")
    logging.info("{:.4f}\n".format(posterior_pred_density))


def main(args):
    baseball_dataset = pd.read_csv(DATA_URL, "\t")
    train, _, player_names = train_test_split(baseball_dataset)
    at_bats, hits = train[:, 0], train[:, 1]
    logging.info("Original Dataset:")
    logging.info(baseball_dataset)

    # (1) Full Pooling Model
    init_params, potential_fn, transforms, _ = initialize_model(fully_pooled, model_args=(at_bats, hits),
                                                                num_chains=args.num_chains)
    nuts_kernel = NUTS(potential_fn=potential_fn)
    mcmc = MCMC(nuts_kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains,
                initial_params=init_params,
                transforms=transforms)
    mcmc.run(at_bats, hits)
    diagnostics = mcmc.diagnostics()
    samples_fully_pooled = mcmc.get_samples()
    logging.info("\nModel: Fully Pooled")
    logging.info("===================")
    logging.info("\nphi:")
    logging.info(summary(samples_fully_pooled,
                         sites=["phi"],
                         player_names=player_names,
                         diagnostics=diagnostics)["phi"])
    num_divergences = sum(map(len, diagnostics["divergences"].values()))
    logging.info("\nNumber of divergent transitions: {}\n".format(num_divergences))
    sample_posterior_predictive(fully_pooled, samples_fully_pooled, baseball_dataset)
    evaluate_log_posterior_density(fully_pooled, samples_fully_pooled, baseball_dataset)

    # (2) No Pooling Model
    init_params, potential_fn, transforms, _ = initialize_model(not_pooled, model_args=(at_bats, hits),
                                                                num_chains=args.num_chains)
    nuts_kernel = NUTS(potential_fn=potential_fn)
    mcmc = MCMC(nuts_kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains,
                initial_params=init_params,
                transforms=transforms)
    mcmc.run(at_bats, hits)
    diagnostics = mcmc.diagnostics()
    samples_not_pooled = mcmc.get_samples()
    logging.info("\nModel: Not Pooled")
    logging.info("=================")
    logging.info("\nphi:")
    logging.info(summary(samples_not_pooled,
                         sites=["phi"],
                         player_names=player_names,
                         diagnostics=diagnostics)["phi"])
    num_divergences = sum(map(len, diagnostics["divergences"].values()))
    logging.info("\nNumber of divergent transitions: {}\n".format(num_divergences))
    sample_posterior_predictive(not_pooled, samples_not_pooled, baseball_dataset)
    evaluate_log_posterior_density(not_pooled, samples_not_pooled, baseball_dataset)

    # (3) Partially Pooled Model
    init_params, potential_fn, transforms, _ = initialize_model(partially_pooled, model_args=(at_bats, hits),
                                                                num_chains=args.num_chains)
    nuts_kernel = NUTS(potential_fn=potential_fn)

    mcmc = MCMC(nuts_kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains,
                initial_params=init_params,
                transforms=transforms)
    mcmc.run(at_bats, hits)
    diagnostics = mcmc.diagnostics()
    samples_partially_pooled = mcmc.get_samples()
    logging.info("\nModel: Partially Pooled")
    logging.info("=======================")
    logging.info("\nphi:")
    logging.info(summary(samples_partially_pooled,
                         sites=["phi"],
                         player_names=player_names,
                         diagnostics=diagnostics)["phi"])
    num_divergences = sum(map(len, diagnostics["divergences"].values()))
    logging.info("\nNumber of divergent transitions: {}\n".format(num_divergences))
    sample_posterior_predictive(partially_pooled, samples_partially_pooled, baseball_dataset)
    evaluate_log_posterior_density(partially_pooled, samples_partially_pooled, baseball_dataset)

    # (4) Partially Pooled with Logit Model
    init_params, potential_fn, transforms, _ = initialize_model(partially_pooled_with_logit,
                                                                model_args=(at_bats, hits),
                                                                num_chains=args.num_chains)
    nuts_kernel = NUTS(potential_fn=potential_fn, transforms=transforms)
    mcmc = MCMC(nuts_kernel,
                num_samples=args.num_samples,
                warmup_steps=args.warmup_steps,
                num_chains=args.num_chains,
                initial_params=init_params,
                transforms=transforms)
    mcmc.run(at_bats, hits)
    diagnostics = mcmc.diagnostics()
    samples_partially_pooled_logit = mcmc.get_samples()
    logging.info("\nModel: Partially Pooled with Logit")
    logging.info("==================================")
    logging.info("\nSigmoid(alpha):")
    logging.info(summary(samples_partially_pooled_logit,
                         sites=["alpha"],
                         player_names=player_names,
                         transforms={"alpha": torch.sigmoid},
                         diagnostics=diagnostics)["alpha"])
    num_divergences = sum(map(len, diagnostics["divergences"].values()))
    logging.info("\nNumber of divergent transitions: {}\n".format(num_divergences))
    sample_posterior_predictive(partially_pooled_with_logit, samples_partially_pooled_logit,
                                baseball_dataset)
    evaluate_log_posterior_density(partially_pooled_with_logit, samples_partially_pooled_logit,
                                   baseball_dataset)


if __name__ == "__main__":
    assert pyro.__version__.startswith('0.3.4')
    parser = argparse.ArgumentParser(description="Baseball batting average using HMC")
    parser.add_argument("-n", "--num-samples", nargs="?", default=200, type=int)
    parser.add_argument("--num-chains", nargs='?', default=4, type=int)
    parser.add_argument("--warmup-steps", nargs='?', default=100, type=int)
    parser.add_argument("--rng_seed", nargs='?', default=0, type=int)
    parser.add_argument("--jit", action="store_true", default=False,
                        help="use PyTorch jit")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="run this example in GPU")
    args = parser.parse_args()

    # work around the error "CUDA error: initialization error" when arg.cuda is False
    # see https://github.com/pytorch/pytorch/issues/2517
    torch.multiprocessing.set_start_method("spawn")
    pyro.set_rng_seed(args.rng_seed)
    # Enable validation checks
    pyro.enable_validation(__debug__)

    # work around with the error "RuntimeError: received 0 items of ancdata"
    # see https://discuss.pytorch.org/t/received-0-items-of-ancdata-pytorch-0-4-0/19823
    torch.multiprocessing.set_sharing_strategy("file_system")

    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    main(args)
