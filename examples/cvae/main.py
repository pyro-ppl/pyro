# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse

import baseline
import cvae
import pandas as pd
import torch
from util import generate_table, get_data, visualize

import pyro


def main(args):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    results = []
    columns = []

    for num_quadrant_inputs in args.num_quadrant_inputs:
        # adds an s in case of plural quadrants
        maybes = "s" if num_quadrant_inputs > 1 else ""

        print(
            "Training with {} quadrant{} as input...".format(
                num_quadrant_inputs, maybes
            )
        )

        # Dataset
        datasets, dataloaders, dataset_sizes = get_data(
            num_quadrant_inputs=num_quadrant_inputs, batch_size=128
        )

        # Train baseline
        baseline_net = baseline.train(
            device=device,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            early_stop_patience=args.early_stop_patience,
            model_path="baseline_net_q{}.pth".format(num_quadrant_inputs),
        )

        # Train CVAE
        cvae_net = cvae.train(
            device=device,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            early_stop_patience=args.early_stop_patience,
            model_path="cvae_net_q{}.pth".format(num_quadrant_inputs),
            pre_trained_baseline_net=baseline_net,
        )

        # Visualize conditional predictions
        visualize(
            device=device,
            num_quadrant_inputs=num_quadrant_inputs,
            pre_trained_baseline=baseline_net,
            pre_trained_cvae=cvae_net,
            num_images=args.num_images,
            num_samples=args.num_samples,
            image_path="cvae_plot_q{}.png".format(num_quadrant_inputs),
        )

        # Retrieve conditional log likelihood
        df = generate_table(
            device=device,
            num_quadrant_inputs=num_quadrant_inputs,
            pre_trained_baseline=baseline_net,
            pre_trained_cvae=cvae_net,
            num_particles=args.num_particles,
            col_name="{} quadrant{}".format(num_quadrant_inputs, maybes),
        )
        results.append(df)
        columns.append("{} quadrant{}".format(num_quadrant_inputs, maybes))

    results = pd.concat(results, axis=1, ignore_index=True)
    results.columns = columns
    results.loc["Performance gap", :] = results.iloc[0, :] - results.iloc[1, :]
    results.to_csv("results.csv")


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.9.1")
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument(
        "-nq",
        "--num-quadrant-inputs",
        metavar="N",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="num of quadrants to use as inputs",
    )
    parser.add_argument(
        "-n", "--num-epochs", default=101, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "-esp", "--early-stop-patience", default=3, type=int, help="early stop patience"
    )
    parser.add_argument(
        "-lr", "--learning-rate", default=1.0e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="whether to use cuda"
    )
    parser.add_argument(
        "-vi",
        "--num-images",
        default=10,
        type=int,
        help="number of images to visualize",
    )
    parser.add_argument(
        "-vs",
        "--num-samples",
        default=10,
        type=int,
        help="number of samples to visualize per image",
    )
    parser.add_argument(
        "-p",
        "--num-particles",
        default=10,
        type=int,
        help="n of particles to estimate logpθ(y|x,z) in ELBO",
    )
    args = parser.parse_args()

    main(args)
