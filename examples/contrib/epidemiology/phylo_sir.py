# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import datetime
import logging
import os
import re

import pandas as pd
import torch
from Bio import Phylo

import pyro
from pyro.contrib.epidemiology import OverdispersedSEIRModel

logging.basicConfig(format='%(message)s', level=logging.INFO)

# The following data files were copied on 2020-05-14 from
# https://github.com/czbiohub/EpiGen-COVID19/tree/master/files
DIRNAME = os.path.dirname(os.path.abspath(__file__))
TIMESERIES_FILE = os.path.join(DIRNAME, "california_timeseries.txt")
TIMETREE_FILE = os.path.join(DIRNAME, "california_timetree.nexus")


def load_data(args):
    # Load time series.
    logging.info("loading {}".format(args.timeseries_file))
    df = pd.read_csv(args.timeseries_file, sep="\t")
    df["date"] = pd.to_datetime(df["date"])
    start_date = df["date"][0]
    new_cases = torch.tensor(df["new_cases"], dtype=torch.float)

    # Load time tree.
    logging.info("loading {}".format(args.timetree_file))
    with open(args.timetree_file) as f:
        for tree in Phylo.parse(f, "nexus"):
            break
    if args.plot:
        # Fix a parsing error for whereby internal nodes interpret .name as
        # .confidence
        for clade in tree.find_clades():
            if clade.confidence:
                clade.name = clade.confidence
                clade.confidence = None
        Phylo.draw(tree, do_show=False)

    # The only data needed from the tree are the times of events.
    leaf_times = []
    coal_times = []
    start_days_after_2020 = (start_date - datetime.datetime(2020, 1, 1, 0, 0)).days
    for clade in tree.find_clades():
        # Parse comments with time in years, e.g. "[&date=2020.16]"
        date_string = re.search(r"date=(\d\d\d\d\.\d\d)", clade.comment).group(1)
        days_after_2020 = (float(date_string) - 2020) * 365.25
        time = days_after_2020 - start_days_after_2020

        # Split nodes into leaves = tips, and binary coalescent events.
        num_children = len(clade)
        if num_children == 0:
            leaf_times.append(time)
        else:
            # Pyro expects binary coalescent events, so we split n-ary events
            # into n-1 separate binary events.
            for _ in range(num_children - 1):
                coal_times.append(time)
    assert len(leaf_times) == 1 + len(coal_times)
    leaf_times = torch.tensor(leaf_times, dtype=torch.float)
    coal_times = torch.tensor(coal_times, dtype=torch.float)

    return new_cases, leaf_times, coal_times


def infer(args, model):
    energies = []

    def hook_fn(kernel, *unused):
        e = float(kernel._potential_energy_last)
        energies.append(e)
        if args.verbose:
            logging.info("potential = {:0.6g}".format(e))

    mcmc = model.fit(heuristic_num_particles=args.num_particles,
                     heuristic_ess_threshold=args.ess_threshold,
                     warmup_steps=args.warmup_steps,
                     num_samples=args.num_samples,
                     max_tree_depth=args.max_tree_depth,
                     num_quant_bins=args.num_bins,
                     dct=args.dct,
                     hook_fn=hook_fn)

    mcmc.summary()
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.plot(energies)
        plt.xlabel("MCMC step")
        plt.ylabel("potential energy")
        plt.title("MCMC energy trace")
        plt.tight_layout()

    return model.samples


def evaluate(args, samples):
    # Print estimated values.
    names = {"basic_reproduction_number": "R0",
             "response_rate": "rho",
             "concentration": "k"}
    for name, key in names.items():
        mean = samples[key].mean().item()
        std = samples[key].std().item()
        logging.info("{}: estimate = {:0.3g} \u00B1 {:0.3g}".format(key, mean, std))

    # Optionally plot histograms.
    if args.plot:
        import matplotlib.pyplot as plt
        import seaborn as sns
        fig, axes = plt.subplots(len(names), 1, figsize=(5, 2.5 * len(names)))
        axes[0].set_title("Posterior parameter estimates")
        for ax, (name, key) in zip(axes, names.items()):
            sns.distplot(samples[key], ax=ax, label="posterior")
            ax.set_xlabel(key + " = " + name.replace("_", " "))
            ax.set_yticks(())
            ax.legend(loc="best")
        plt.tight_layout()


def predict(args, model):
    samples = model.predict(forecast=args.forecast)

    obs = model.data

    new_I = samples.get("S2I", samples.get("E2I"))
    median = new_I.median(dim=0).values
    logging.info("Median prediction of new infections (starting on day 0):\n{}"
                 .format(" ".join(map(str, map(int, median)))))

    # Optionally plot the latent and forecasted series of new infections.
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        time = torch.arange(args.duration + args.forecast)
        p05 = new_I.kthvalue(int(round(0.5 + 0.05 * args.num_samples)), dim=0).values
        p95 = new_I.kthvalue(int(round(0.5 + 0.95 * args.num_samples)), dim=0).values
        plt.fill_between(time, p05, p95, color="red", alpha=0.3, label="90% CI")
        plt.plot(time, median, "r-", label="median")
        plt.plot(time[:args.duration], obs, "k.", label="observed")
        plt.axvline(args.duration - 0.5, color="gray", lw=1)
        plt.xlim(0, len(time) - 1)
        plt.ylim(0, None)
        plt.xlabel("day after first infection")
        plt.ylabel("new infections per day")
        plt.title("New infections in population of {}".format(args.population))
        plt.legend(loc="upper left")
        plt.tight_layout()


def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.rng_seed)

    # Load data.
    new_cases, leaf_times, coal_times = load_data(args)
    args.duration = len(new_cases)
    logging.info("Observed {:d} infections:\n{}".format(
        int(new_cases.sum().item()), " ".join(str(int(x)) for x in new_cases)))

    # Run inference.
    model = OverdispersedSEIRModel(args.population, args.incubation_time,
                                   args.recovery_time, new_cases,
                                   leaf_times=leaf_times, coal_times=coal_times)
    samples = infer(args, model)

    # Evaluate fit.
    evaluate(args, samples)

    # Predict latent time series.
    if args.forecast:
        predict(args, model)


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.3.1')
    parser = argparse.ArgumentParser(
        description="Compartmental epidemiology modeling using HMC")
    parser.add_argument("--timeseries-file", default=TIMESERIES_FILE)
    parser.add_argument("--timetree-file", default=TIMETREE_FILE)
    parser.add_argument("-p", "--population", default=40000000, type=int)
    parser.add_argument("-f", "--forecast", default=10, type=int)
    parser.add_argument("-tau", "--recovery-time", default=14.0, type=float)
    parser.add_argument("-e", "--incubation-time", default=5.5, type=float)
    parser.add_argument("--dct", type=float,
                        help="smoothing for discrete cosine reparameterizer")
    parser.add_argument("-n", "--num-samples", default=200, type=int)
    parser.add_argument("-np", "--num-particles", default=1024, type=int)
    parser.add_argument("-ess", "--ess-threshold", default=0.5, type=float)
    parser.add_argument("-w", "--warmup-steps", default=100, type=int)
    parser.add_argument("-t", "--max-tree-depth", default=5, type=int)
    parser.add_argument("-r", "--rng-seed", default=0, type=int)
    parser.add_argument("-nb", "--num-bins", default=4, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    main(args)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.show()
