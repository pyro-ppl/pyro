from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch
from observations import boston_housing

import pyro
from pyro.contrib.examples.util import get_data_directory
from pyro.contrib.tabular import Boolean, Real, TreeCat
from pyro.contrib.tabular.treecat import print_tree
from pyro.optim import Adam


def load_data():
    x_train, metadata = boston_housing(get_data_directory(__file__))
    x_train = torch.tensor(x_train.T, dtype=torch.get_default_dtype()).contiguous()
    features = []
    data = []
    logging.info("loaded {} rows x {} features:".format(x_train.size(1), x_train.size(0)))
    for name, column in zip(metadata["columns"], x_train):
        ftype = Boolean if name == "CHAS" else Real
        features.append(ftype(name))
        data.append(column)
        logging.info(" {} {}".format(name, ftype.__name__))
    return features, data


def partition_data(data, target_size):
    num_rows = len(data[0])
    begin = 0
    while begin < num_rows:
        end = begin + target_size
        yield [col[begin: end] for col in data]
        begin = end


def main(args):
    features, data = load_data()

    # Train a model.
    pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)
    model = TreeCat(features, args.capacity, annealing_rate=args.learning_rate)
    optim = Adam({"lr": args.learning_rate})
    trainer = model.trainer(optim, backend=args.backend)
    trainer.init(data)
    num_rows = len(data[0])
    for epoch in range(args.num_epochs):
        loss = 0
        for batch in partition_data(data, args.batch_size):
            loss += trainer.step(batch, num_rows=num_rows)
        logging.info("epoch {} loss = {:0.4g}".format(epoch, loss))
    logging.info("Learned Tree:\n{}".format(print_tree(model.edges, model.features)))

    # Predict housing price (MEDV) as function of NOX.
    index = {f.name: i for i, f in enumerate(features)}
    nox = data[index["NOX"]]
    nox = torch.linspace(nox.min(), nox.max(), 20)
    query = [None] * len(features)
    query[index["NOX"]] = nox
    samples = model.impute(query, num_samples=args.num_samples)
    price = samples[index["MEDV"]]  # = median housing value / $1000
    assert price.shape == (args.num_samples, 20)
    price_mean = price.mean(dim=0)
    price_std = price.std(dim=0)
    logging.info("--------------------------------------------")
    logging.info("Predicted housing price as a function of NOX")
    logging.info("--------------------------------------------")
    logging.info("NOX\tMEDV")
    for i in range(20):
        logging.info("{:0.3g}\t{:0.3g} += {:0.3g}".format(
            nox[i].item(), price_mean[i].item(), price_std[i].item()))
    logging.info("--------------------------------------------")


if __name__ == "__main__":
    assert pyro.__version__.startswith('0.3.3')
    parser = argparse.ArgumentParser(description="Tabular data analysis of Boston Housing")
    parser.add_argument("-c", "--capacity", default=16, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.05, type=float)
    parser.add_argument("-b", "--batch-size", default=22, type=int)
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-s", "--num-samples", default=100, type=int)
    parser.add_argument("--backend", default="python")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(format="%(relativeCreated) 9d %(message)s",
                        level=logging.DEBUG if args.verbose else logging.INFO)
    main(args)
