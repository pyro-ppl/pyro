# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import numpy as np
import torch

import pyro
import pyro.distributions as dist
from pyro.contrib.examples.bart import load_bart_od
from pyro.contrib.forecast import ForecastingModel, backtest
from pyro.ops.tensor_utils import periodic_cumsum, periodic_repeat

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)


def preprocess(args):
    """
    Extract a tensor of (arrivals,departures) to Embarcadero station.
    """
    print("Loading data")
    dataset = load_bart_od()
    i = dataset["stations"].index("EMBR")
    arrivals = dataset["counts"][:, :, i].sum(-1)
    departures = dataset["counts"][:, i, :].sum(-1)
    data = torch.stack([arrivals, departures], dim=-1)
    covariates = torch.zeros(len(data), 0)  # Has correct len() but no elements.
    return data, covariates


# We define a model by subclassing the ForecastingModel class and implementing
# a single .model() method.
class Model(ForecastingModel):
    # The .model() method inputs two tensors: a fake tensor zero_data that is
    # the same size and dtype as the real data (but of course the generative
    # model shouldn't depend on the value of the data it generates!), and a
    # tensor of covariates. Our simple model depends on no covariates, so we
    # simply pass in an empty tensor (see  the preprocess() function above).
    def model(self, zero_data, covariates):
        duration = zero_data.size(-2)
        period = 24 * 7

        # Sample global parameters.
        noise_scale = pyro.sample("noise_scale",
                                  dist.LogNormal(torch.full((2,), -3), 1).to_event(1))
        trans_timescale = pyro.sample("trans_timescale",
                                      dist.LogNormal(torch.zeros(2), 1).to_event(1))

        trans_scale = pyro.sample("trans_scale",
                                  dist.LogNormal(torch.zeros(2), 0.1).to_event(1))
        trans_corr = pyro.sample("trans_corr",
                                 dist.LKJCorrCholesky(2, torch.ones(())))
        trans_scale_tril = trans_scale.unsqueeze(-1) * trans_corr

        obs_scale = pyro.sample("obs_scale",
                                dist.LogNormal(torch.zeros(2), 0.1).to_event(1))
        obs_corr = pyro.sample("obs_corr",
                               dist.LKJCorrCholesky(2, torch.ones(())))
        obs_scale_tril = obs_scale.unsqueeze(-1) * obs_corr

        # Note the initial seasonality should be sampled in a plate with the
        # same dim as the time_plate, dim=-1. That way we can repeat the dim
        # below using periodic_repeat().
        with pyro.plate("season_plate", period,  dim=-1):
            season_init = pyro.sample("season_init",
                                      dist.Normal(torch.zeros(2), 1).to_event(1))

        # Sample independent noise at each time step.
        with self.time_plate:
            season_noise = pyro.sample("season_noise",
                                       dist.Normal(0, noise_scale).to_event(1))

        # Construct a prediction. This prediction has an exactly repeated
        # seasonal part plus slow seasonal drift. We use two deterministic,
        # linear functions to transform our diagonal Normal noise to nontrivial
        # samples from a Gaussian process.
        prediction = (periodic_repeat(season_init, duration, dim=-2) +
                      periodic_cumsum(season_noise, period, dim=-2))

        # Construct a joint noise model. This model is a GaussianHMM, whose
        # .rsample() and .log_prob() methods are parallelized over time; this
        # this entire model is parallelized over time.
        init_dist = dist.Normal(torch.zeros(2), 100).to_event(1)
        trans_mat = trans_timescale.neg().exp().diag_embed()
        trans_dist = dist.MultivariateNormal(torch.zeros(2), scale_tril=trans_scale_tril)
        obs_mat = torch.eye(2)
        obs_dist = dist.MultivariateNormal(torch.zeros(2), scale_tril=obs_scale_tril)
        noise_model = dist.GaussianHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist,
                                       duration=duration)

        # The final statement registers our noise model and prediction.
        self.predict(noise_model, prediction)


def main(args):
    pyro.enable_validation(__debug__)
    data, covariates = preprocess(args)

    # We will model positive count data by log1p-transforming it into real
    # valued data.  But since we want to evaluate back in the count domain, we
    # will also define a transform to apply during evaluation, transforming
    # from real back to count-valued data. Truth is mapped by the log1p()
    # inverse expm1(), but the prediction will be sampled from a Poisson
    # distribution.
    data = data.log1p()

    def transform(pred, truth):
        pred = torch.poisson(pred.clamp(min=1e-4).expm1())
        truth = truth.expm1()
        return pred, truth

    # The backtest() function automatically trains and evaluates our model on
    # different windows of data.
    forecaster_options = {
        "num_steps": args.num_steps,
        "learning_rate": args.learning_rate,
        "log_every": args.log_every,
    }
    metrics = backtest(data, covariates, Model(),
                       train_window=args.train_window,
                       test_window=args.test_window,
                       stride=args.stride,
                       num_samples=args.num_samples,
                       forecaster_options=forecaster_options)

    for name in ["mae", "rmse", "crps"]:
        values = [m[name] for m in metrics]
        mean = np.mean(values)
        std = np.std(values)
        print("{} = {:0.3g} +- {:0.3g}".format(name, mean, std))
    return metrics


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.2.1')
    parser = argparse.ArgumentParser(description="Causal Effect Variational Autoencoder")
    parser.add_argument("--train-window", default=2160, type=int)
    parser.add_argument("--test-window", default=336, type=int)
    parser.add_argument("--stride", default=168, type=int)
    parser.add_argument("-n", "--num-steps", default=501, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("--num-samples", default=100, type=int)
    parser.add_argument("--log-every", default=50, type=int)
    parser.add_argument("--seed", default=1234567890, type=int)
    args = parser.parse_args()
    main(args)
