from __future__ import absolute_import, division, print_function

import argparse
import logging

import torch
from observations import boston_housing

import pyro
from pyro.contrib.examples.util import get_data_directory
from pyro.contrib.tabular import Boolean, Real, TreeCat, TreeCatTrainer
from pyro.optim import Adam

logging.basicConfig(format="%(relativeCreated) 9d %(message)s", level=logging.INFO)


def load_data():
    x_train, metadata = boston_housing(get_data_directory(__file__))
    x_train = torch.tensor(x_train.T, dtype=torch.get_default_dtype()).contiguous()
    features = []
    data = []
    for name, column in zip(metadata["columns"], x_train):
        ftype = Boolean if name == "CHAR" else Real
        features.append(ftype(name))
        data.append(column)
    logging.info("loaded {} rows x {} features".format(x_train.size(1), x_train.size(0)))
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
    pyro.enable_validation(args.validate)
    model = TreeCat(features, args.capacity)
    optim = Adam({"lr": args.learning_rate})
    trainer = TreeCatTrainer(model, optim)
    num_rows = len(data[0])
    for epoch in range(args.num_epochs):
        loss = 0
        for batch in partition_data(data, args.batch_size):
            loss += trainer.step(data, num_rows=num_rows)
        logging.info("epoch {} loss = {:0.4g}".format(epoch, loss))

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
    logging.info("--------------------------------")
    logging.info("NOX\tMEDV")
    for i in range(20):
        logging.info("{:0.3g}\t{:0.3g} += {:0.3g}".format(
            nox[i].item(), price_mean[i].item(), price_std[i].item()))
    logging.info("--------------------------------")


if __name__ == "__main__":
    assert pyro.__version__.startswith('0.3.1')
    parser = argparse.ArgumentParser(description="Tabular data analysis of Boston Housing")
    parser.add_argument("-c", "--capacity", default=16, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.002, type=float)
    parser.add_argument("-b", "--batch-size", default=22, type=int)
    parser.add_argument("-n", "--num-epochs", default=100, type=int)
    parser.add_argument("-s", "--num-samples", default=100, type=int)
    parser.add_argument("-v", "--validate", default=True, type=bool)
    args = parser.parse_args()
    main(args)
