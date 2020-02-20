# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import numpy as np
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.contrib.examples.bart import load_bart_od
from pyro.contrib.forecast import ForecastingModel, backtest
from pyro.infer.reparam import DiscreteCosineReparam
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
    data = torch.stack([arrivals, departures], dim=-1).log1p()
    covariates = torch.zeros(len(data), 0)
    return data, covariates


class Model(ForecastingModel):
    def model(self, zero_data, covariates):
        duration = zero_data.size(-2)
        period = 24 * 7

        # Sample global parameters.
        noise_scale = pyro.sample("noise_scale",
                                  dist.LogNormal(torch.full((2,), -3), 1).to_event(1))
        trans_mat_noise = pyro.sample("trans_mat_noise",
                                      dist.Normal(torch.zeros(2, 2), 0.1).to_event(2))
        trans_dist_scale = pyro.sample("trans_dist_scale",
                                       dist.LogNormal(torch.zeros(2), 0.1).to_event(1))
        obs_dist_scale = pyro.sample("obs_dist_scale",
                                     dist.LogNormal(torch.zeros(2), 0.1).to_event(1))
        with pyro.plate("season_plate", period,  dim=-1):
            season_init = pyro.sample("season_init",
                                      dist.Normal(torch.zeros(2), 1).to_event(1))

        # Sample independent noise at each time step.
        with self.time_plate:
            season_noise = pyro.sample("season_noise",
                                       dist.Normal(0, noise_scale).to_event(1))

        # TODO get DiscreteCosineReparam working.
        # with poutine.reparam(config={"season_noise": DiscreteCosineReparam(dim=-2)}):
        #     season_noise = pyro.sample("season_noise",
        #                                dist.Normal(torch.zeros(duration, 2),
        #                                            noise_scale.unsqueeze(-2)).to_event(2))
        # if season_noise.dim() > 3:
        #     season_noise = season_noise.squeeze(-3)

        # Construct prediction.
        prediction = (periodic_repeat(season_init, duration, dim=-2) +
                      periodic_cumsum(season_noise, period, dim=-2))

        # Construct a joint noise model.
        noise_model = dist.GaussianHMM(
            initial_dist=dist.Normal(torch.zeros(2), 100).to_event(1),
            transition_matrix=torch.eye(2) + trans_mat_noise,
            transition_dist=dist.Normal(0, trans_dist_scale).to_event(1),
            observation_matrix=torch.eye(2),
            observation_dist=dist.Normal(0, obs_dist_scale).to_event(1),
            duration=duration)

        self.predict(noise_model, prediction)


def main(args):
    pyro.enable_validation(__debug__)

    data, covariates = preprocess(args)

    def transform(pred, truth):
        pred = torch.poisson(pred.clamp(min=1e-4).expm1())
        truth = truth.expm1()
        return pred, truth

    forecaster_options = {
        "num_steps": args.num_steps,
        "learning_rate": args.learning_rate,
        "log_every": args.log_every,
    }
    metrics = backtest(data, covariates, Model(),
                       train_window=args.train_window,
                       test_window=args.test_window,
                       stride=args.stride,
                       forecaster_options=forecaster_options)

    for name in ["mae", "rmse", "crps"]:
        values = [m[name] for m in metrics]
        mean = np.mean(values)
        std = np.std(values)
        print("{} = {:0.3g} +- {:0.3g}".format(mean, std))
    return metrics


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.2.1')
    parser = argparse.ArgumentParser(description="Causal Effect Variational Autoencoder")
    parser.add_argument("--train-window", default=5040, type=int)
    parser.add_argument("--test-window", default=336, type=int)
    parser.add_argument("--stride", default=168, type=int)
    parser.add_argument("-n", "--num-steps", default=501, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("--log-every", default=50, type=float)
    parser.add_argument("--seed", default=1234567890, type=int)
    args = parser.parse_args()
    main(args)
