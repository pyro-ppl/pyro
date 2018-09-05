import argparse
import torch
from torch.distributions.transforms import AffineTransform, SigmoidTransform
import numpy as np
import matplotlib.pyplot as plt

import pyro
from pyro import optim
import pyro.distributions as dist
from pyro.contrib.oed.eig import barber_agakov_ape
from pyro.contrib.oed.util import rmv

from models.bayes_linear import sigmoid_model, rf_group_assignments
from guides.amort import SigmoidGuide

# Random effects designs
AB_test_reff_6d_10n_12p, AB_sigmoid_design_6d = rf_group_assignments(10)

sigmoid_ba_guide = lambda d: SigmoidGuide(d, 10, {"w1": 2, "w2": 10})


def true_model(design):
    w1 = torch.tensor([-1., 1.])
    w2 = torch.tensor([-.5, .5, -.5, .5, -.5, 2., -2., 2., -2., 0.])
    w = torch.cat([w1, w2], dim=-1)
    k = torch.tensor(.1)
    response_mean = rmv(design, w)

    base_dist = dist.Normal(response_mean, torch.tensor(1.)).independent(1)
    k = k.expand(response_mean.shape)
    transforms = [AffineTransform(loc=0., scale=k), SigmoidTransform()]
    response_dist = dist.TransformedDistribution(base_dist, transforms)
    return pyro.sample("y", response_dist)


def iqr(array):
    return np.percentile(array, 75) - np.percentile(array, 25)


def main(num_experiments, num_runs):

    results = {'oed': [], 'rand': []}

    for typ in ['oed', 'rand']:
        print("Type", typ)

        for k in range(1, num_runs+1):
            print("Run", k)

            model = sigmoid_model(torch.tensor(0.),
                                  torch.tensor([10., 2.5]),
                                  torch.tensor(0.),
                                  torch.tensor([1.]*5 + [10.]*5),
                                  torch.tensor(1.),
                                  100.*torch.ones(10),
                                  1000.*torch.ones(10),
                                  AB_sigmoid_design_6d)
            my_guide = sigmoid_ba_guide(6)
            ba_kwargs = {"num_samples": 100, "num_steps": 500, "guide": my_guide.guide,
                         "optim": optim.Adam({"lr": 0.05}), "final_num_samples": 500}

            for experiment_number in range(1, num_experiments+1):
                pyro.clear_param_store()

                estimation_surface = barber_agakov_ape(model, AB_test_reff_6d_10n_12p, "y", "w1", **ba_kwargs)

                # Run experiment
                if typ == 'oed':
                    d_star_index = torch.argmin(estimation_surface)
                elif typ == 'rand':
                    d_star_index = torch.randint(6, tuple())
                d_star_index = int(d_star_index)
                design = AB_test_reff_6d_10n_12p[d_star_index, ...]
                y = true_model(design)
                mu_dict, scale_tril_dict = my_guide({"y": y}, AB_test_reff_6d_10n_12p, ["w1"])
                mu, scale_tril = mu_dict["w1"], scale_tril_dict["w1"]

                model = sigmoid_model(mu[d_star_index, ...].detach(),
                                      torch.diag(scale_tril[d_star_index, ...].detach()),
                                      torch.tensor(0.),
                                      torch.tensor([1.]*5 + [10.]*5),
                                      torch.tensor(1.),
                                      100.*torch.ones(10),
                                      1000.*torch.ones(10),
                                      AB_sigmoid_design_6d)

            results[typ].append((mu[d_star_index, ...].detach().numpy(),
                                 scale_tril[d_star_index, ...].detach().numpy()))

    # Box-and-whisker plot of final entropy
    covs = {k: [x[1].transpose().dot(x[1]) for x in v] for k, v in results.items()}
    entropies = {k: np.array([0.5*np.linalg.slogdet(2*np.pi*np.e*cov)[1] for cov in v]) for k, v in covs.items()}
    means = np.array([entropies['oed'].mean(), entropies['rand'].mean()])
    iqrs = np.array([iqr(entropies['oed']), iqr(entropies['rand'])])
    plt.figure()
    x = np.array([0., 1.])
    plt.bar(x, height=means, yerr=iqrs, tick_label=['OED', 'Random design'],
            color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0:2],
            capsize=10)
    plt.ylabel("Final posterior entropy")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sigmoid iterated experiment design")
    parser.add_argument("--num-experiments", nargs="?", default=5, type=int)
    parser.add_argument("--num-runs", nargs="?", default=5, type=int)
    args = parser.parse_args()
    main(args.num_experiments, args.num_runs)
