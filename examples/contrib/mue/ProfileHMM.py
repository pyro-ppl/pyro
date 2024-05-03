# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
A standard profile HMM model [1], which corresponds to a constant (delta
function) distribution with a MuE observation [2]. This is a standard
generative model of variable-length biological sequences (e.g. proteins) which
does not require preprocessing the data by building a multiple sequence
alignment. It can be compared to a more complex MuE model in this package,
the FactorMuE.

An example dataset consisting of proteins similar to the human papillomavirus E6
protein, collected from a non-redundant sequence dataset using jackhmmer, can
be found at
https://github.com/debbiemarkslab/MuE/blob/master/models/examples/ve6_full.fasta

Example run:
python ProfileHMM.py -f PATH/ve6_full.fasta -b 10 -M 174 --indel-prior-bias 10.
    -e 15 -lr 0.01 --jit --cuda
This should take about 9 minutes to run on a GPU. The perplexity should be
around 6.

References:
[1] R. Durbin, S. R. Eddy, A. Krogh, and G. Mitchison (1998)
"Biological sequence analysis: probabilistic models of proteins and nucleic
acids"
Cambridge university press

[2] E. N. Weinstein, D. S. Marks (2021)
"A structured observation distribution for generative biological sequence
prediction and forecasting"
https://www.biorxiv.org/content/10.1101/2020.07.31.231381v2.full.pdf
"""

import argparse
import datetime
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam

import pyro
from pyro.contrib.mue.dataloaders import BiosequenceDataset
from pyro.contrib.mue.models import ProfileHMM
from pyro.optim import MultiStepLR


def generate_data(small_test, include_stop, device):
    """Generate mini example dataset."""
    if small_test:
        mult_dat = 1
    else:
        mult_dat = 10

    seqs = ["BABBA"] * mult_dat + ["BAAB"] * mult_dat + ["BABBB"] * mult_dat
    dataset = BiosequenceDataset(
        seqs, "list", "AB", include_stop=include_stop, device=device
    )

    return dataset


def main(args):
    pyro.set_rng_seed(args.rng_seed)

    # Load dataset.
    if args.cpu_data or not args.cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    if args.test:
        dataset = generate_data(args.small, args.include_stop, device)
    else:
        dataset = BiosequenceDataset(
            args.file,
            "fasta",
            args.alphabet,
            include_stop=args.include_stop,
            device=device,
        )
    args.batch_size = min([dataset.data_size, args.batch_size])
    if args.split > 0.0:
        # Train test split.
        heldout_num = int(np.ceil(args.split * len(dataset)))
        data_lengths = [len(dataset) - heldout_num, heldout_num]
        # Specific data split seed, for comparability across models and
        # parameter initializations.
        pyro.set_rng_seed(args.rng_data_seed)
        indices = torch.randperm(sum(data_lengths), device=device).tolist()
        dataset_train, dataset_test = [
            torch.utils.data.Subset(dataset, indices[(offset - length) : offset])
            for offset, length in zip(np.cumsum(data_lengths), data_lengths)
        ]
    else:
        dataset_train = dataset
        dataset_test = None

    # Construct model.
    latent_seq_length = args.latent_seq_length
    if latent_seq_length is None:
        latent_seq_length = int(dataset.max_length * 1.1)
    model = ProfileHMM(
        latent_seq_length,
        dataset.alphabet_length,
        prior_scale=args.prior_scale,
        indel_prior_bias=args.indel_prior_bias,
        cuda=args.cuda,
        pin_memory=args.pin_mem,
    )

    # Infer with SVI.
    scheduler = MultiStepLR(
        {
            "optimizer": Adam,
            "optim_args": {"lr": args.learning_rate},
            "milestones": json.loads(args.milestones),
            "gamma": args.learning_gamma,
        }
    )
    n_epochs = args.n_epochs
    losses = model.fit_svi(dataset, n_epochs, args.batch_size, scheduler, args.jit)

    # Evaluate.
    train_lp, test_lp, train_perplex, test_perplex = model.evaluate(
        dataset_train, dataset_test, args.jit
    )
    print("train logp: {} perplex: {}".format(train_lp, train_perplex))
    print("test logp: {} perplex: {}".format(test_lp, test_perplex))

    # Plots.
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not args.no_plots:
        plt.figure(figsize=(6, 6))
        plt.plot(losses)
        plt.xlabel("step")
        plt.ylabel("loss")
        if not args.no_save:
            plt.savefig(
                os.path.join(
                    args.out_folder, "ProfileHMM_plot.loss_{}.pdf".format(time_stamp)
                )
            )

        plt.figure(figsize=(6, 6))
        insert = pyro.param("insert_q_mn").detach()
        insert_expect = torch.exp(insert - insert.logsumexp(-1, True))
        plt.plot(insert_expect[:, :, 1].cpu().numpy())
        plt.xlabel("position")
        plt.ylabel("probability of insert")
        plt.legend([r"$r_0$", r"$r_1$", r"$r_2$"])
        if not args.no_save:
            plt.savefig(
                os.path.join(
                    args.out_folder,
                    "ProfileHMM_plot.insert_prob_{}.pdf".format(time_stamp),
                )
            )
        plt.figure(figsize=(6, 6))
        delete = pyro.param("delete_q_mn").detach()
        delete_expect = torch.exp(delete - delete.logsumexp(-1, True))
        plt.plot(delete_expect[:, :, 1].cpu().numpy())
        plt.xlabel("position")
        plt.ylabel("probability of delete")
        plt.legend([r"$u_0$", r"$u_1$", r"$u_2$"])
        if not args.no_save:
            plt.savefig(
                os.path.join(
                    args.out_folder,
                    "ProfileHMM_plot.delete_prob_{}.pdf".format(time_stamp),
                )
            )

    if not args.no_save:
        pyro.get_param_store().save(
            os.path.join(
                args.out_folder, "ProfileHMM_results.params_{}.out".format(time_stamp)
            )
        )
        with open(
            os.path.join(
                args.out_folder,
                "ProfileHMM_results.evaluation_{}.txt".format(time_stamp),
            ),
            "w",
        ) as ow:
            ow.write("train_lp,test_lp,train_perplex,test_perplex\n")
            ow.write(
                "{},{},{},{}\n".format(train_lp, test_lp, train_perplex, test_perplex)
            )
        with open(
            os.path.join(
                args.out_folder, "ProfileHMM_results.input_{}.txt".format(time_stamp)
            ),
            "w",
        ) as ow:
            ow.write("[args]\n")
            args.latent_seq_length = model.latent_seq_length
            for elem in list(args.__dict__.keys()):
                ow.write("{} = {}\n".format(elem, args.__getattribute__(elem)))
            ow.write("alphabet_str = {}\n".format("".join(dataset.alphabet)))
            ow.write("max_length = {}\n".format(dataset.max_length))


if __name__ == "__main__":
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Profile HMM model.")
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Run with generated example dataset.",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        default=False,
        help="Run with small example dataset.",
    )
    parser.add_argument("-r", "--rng-seed", default=0, type=int)
    parser.add_argument("--rng-data-seed", default=0, type=int)
    parser.add_argument(
        "-f", "--file", default=None, type=str, help="Input file (fasta format)."
    )
    parser.add_argument(
        "-a",
        "--alphabet",
        default="amino-acid",
        help="Alphabet (amino-acid OR dna OR ATGC ...).",
    )
    parser.add_argument("-b", "--batch-size", default=10, type=int, help="Batch size.")
    parser.add_argument(
        "-M",
        "--latent-seq-length",
        default=None,
        type=int,
        help="Latent sequence length.",
    )
    parser.add_argument(
        "--include-stop",
        default=False,
        action="store_true",
        help="Include stop symbol at the end of each sequence.",
    )
    parser.add_argument(
        "--prior-scale",
        default=1.0,
        type=float,
        help="Prior scale parameter (all parameters).",
    )
    parser.add_argument(
        "--indel-prior-bias",
        default=10.0,
        type=float,
        help="Indel prior bias parameter.",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=0.001,
        type=float,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--milestones",
        default="[]",
        type=str,
        help="Milestones for multistage learning rate.",
    )
    parser.add_argument(
        "--learning-gamma",
        default=0.5,
        type=float,
        help="Gamma parameter for multistage learning rate.",
    )
    parser.add_argument(
        "-e", "--n-epochs", default=10, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--no-plots", default=False, action="store_true", help="Make plots."
    )
    parser.add_argument(
        "--no-save",
        default=False,
        action="store_true",
        help="Do not save plots and results.",
    )
    parser.add_argument(
        "-outf", "--out-folder", default=".", help="Folder to save plots."
    )
    parser.add_argument(
        "--split",
        default=0.2,
        type=float,
        help=("Fraction of dataset to holdout for testing"),
    )
    parser.add_argument(
        "--jit", default=False, action="store_true", help="JIT compile the ELBO."
    )
    parser.add_argument("--cuda", default=False, action="store_true", help="Use GPU.")
    parser.add_argument(
        "--cpu-data",
        default=False,
        action="store_true",
        help="Keep data on CPU (for large datasets).",
    )
    parser.add_argument(
        "--pin-mem",
        default=False,
        action="store_true",
        help="Use pin_memory for faster GPU transfer.",
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    if args.cuda:
        torch.set_default_device("cuda")

    main(args)
