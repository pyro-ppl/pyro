import numpy as np
import torch

import pyro
from pyro.contrib.timeseries import IndependentMaternGP

import argparse
from os.path import exists
from urllib.request import urlopen


pyro.enable_validation(__debug__)


# download dataset from UCI archive
def download_data():
    if not exists("eeg.dat"):
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff"
        with open("eeg.dat", "wb") as f:
            f.write(urlopen(url).read())


def main(args):
    # download and pre-process EEG data if not in test mode
    if not args.test:
        download_data()
        N_test = 349
        data = np.loadtxt('eeg.dat', delimiter=',', skiprows=19)
        print("[raw data shape] {}".format(data.shape))
        data = torch.tensor(data[::20, :-1]).double()
        print("[data shape after thinning] {}".format(data.shape))
    # in test mode (for continuous integration on github) so create fake data
    else:
        data = torch.randn(20, 3).double()
        N_test = 10

    T, obs_dim = data.shape
    N_train = T - N_test

    # standardize data
    data_mean = data[0:N_train, :].mean(0)
    data -= data_mean
    data_std = data[0:N_train, :].std(0)
    data /= data_std

    torch.manual_seed(args.seed)

    # set up model
    gp = IndependentMaternGP(nu=1.5, obs_dim=obs_dim,
                             log_length_scale_init=0.5 * torch.ones(obs_dim)).double()

    # set up optimizer
    adam = torch.optim.Adam(gp.parameters(), lr=args.init_learning_rate, betas=(args.beta1, 0.999), amsgrad=True)
    gamma = (args.final_learning_rate / args.init_learning_rate) ** (1.0 / args.num_steps)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(adam, gamma=gamma)

    report_frequency = 10

    # training loop
    for step in range(args.num_steps):
        loss = -gp.log_prob(data[0:N_train, :]).sum() / N_train
        loss.backward()
        adam.step()
        scheduler.step()

        if step % report_frequency == 0 or step == args.num_steps - 1:
            print("[step %03d]  loss: %.3f" % (step, loss.item()))

    # plot predictions for three output dimensions
    if args.plot:
        assert not args.test

        # do rolling prediction
        pred_means, pred_stds = np.zeros((N_test, obs_dim)), np.zeros((N_test, obs_dim))
        for t in range(N_test):
            # predict one step into the future, conditioning on all previous data
            pred_dist = gp.predict(data[0:N_train + t, :], torch.tensor([1.0]).double())
            pred_means[t, :] = pred_dist.loc.data.numpy()
            pred_stds[t, :] = pred_dist.scale.data.numpy()

        import matplotlib
        matplotlib.use('Agg')  # noqa: E402
        import matplotlib.pyplot as plt

        f, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        T = data.size(0)
        to_seconds = 117.0 / T

        for k, ax in enumerate(axes):
            which = [0, 4, 10][k]
            ax.plot(to_seconds * np.arange(T), data[:, which], 'ko', markersize=2)
            ax.plot(to_seconds * (N_train + np.arange(N_test)),
                    pred_means[:, which], ls='solid', color='b')
            # plot 90% confidence intervals
            ax.fill_between(to_seconds * (N_train + np.arange(N_test)),
                            pred_means[:, which] - 1.645 * pred_stds[:, which],
                            pred_means[:, which] + 1.645 * pred_stds[:, which],
                            color='blue', alpha=0.20)
            ax.set_ylabel("$y_{%d}$" % (which + 1), fontsize=20)
            ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout(pad=0.7)
        plt.savefig('eeg.pdf')


if __name__ == '__main__':
    assert pyro.__version__.startswith('0.5.1')
    parser = argparse.ArgumentParser(description="contrib.timeseries example usage")
    parser.add_argument("-n", "--num-steps", default=500, type=int)
    parser.add_argument("-s", "--seed", default=0, type=int)
    parser.add_argument("-ilr", "--init-learning-rate", default=0.01, type=float)
    parser.add_argument("-flr", "--final-learning-rate", default=0.001, type=float)
    parser.add_argument("-b1", "--beta1", default=0.50, type=float)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--plot", action='store_true')
    args = parser.parse_args()

    main(args)
