from __future__ import absolute_import, division, print_function

import argparse
import torch
from torch.distributions.transforms import AffineTransform, SigmoidTransform
import numpy as np

import pyro
from pyro import optim
import pyro.distributions as dist
from pyro.contrib.oed.eig import barber_agakov_ape
from pyro.contrib.util import rmv

from pyro.contrib.glmm import sigmoid_model, rf_group_assignments
from pyro.contrib.glmm.guides import SigmoidGuide

"""
Sequential optimal experiment design using a sigmoid-transformed linear model.

In this example, we demonstrate the effectiveness of using optimal experiment design (OED)
over a sequence of experiments.

In many settings, the response in a study is restricted to fall in a bounded interval,
say :math:`[0,1]`. For instance, in a psychology study which asks participants to respond
with a slider. To adapt a linear model to this case, we add a sigmoid transformation
using the function

    :math:`\frac{1}{1 + e^{-x}}`

to the output from a regular linear model. To make the model more realistic, we allow
each participant a random offset and slope to account for personal differences in using the slider.

The experiment we are designing is an AB test. We choose the allocation of participants
to the groups A and B using OED, in this case by maximizing the Barber-Agakov bound on
expected information gain (EIG).

Having found the optimal experiment, we produce data from a fixed simulator that represents
the actual responses of the respondents. We update our beliefs about the effects of A and B
using Bayesian inference (which can actually be derived from the posterior approximation found
when maximizing the Barber-Agakov bound).

We then design and perform the next experiment, assuming that we have a fresh pool of
participants.

To assess the benefit of using OED, rather than assigning participants at random to the groups,
we examine the posterior entropy of the final posterior. This tells us how certain (or not)
we have become about the effects of A and B. We typically find lower entropy (lower uncertainty)
when using OED.
"""

# Random effects designs
AB_test_reff_6d_10n_12p, AB_sigmoid_design_6d = rf_group_assignments(10)

sigmoid_ba_guide = lambda d: SigmoidGuide(d, 10, {"w1": 2, "w2": 10})  # noqa: E731


def true_model(design):
    w1 = torch.tensor([-1., 1.])
    w2 = torch.tensor([-.5, .5, -.5, .5, -.5, 2., -2., 2., -2., 0.])
    w = torch.cat([w1, w2], dim=-1)
    k = torch.tensor(.1)
    response_mean = rmv(design, w)

    base_dist = dist.Normal(response_mean, torch.tensor(1.)).to_event(1)
    k = k.expand(response_mean.shape)
    transforms = [AffineTransform(loc=0., scale=k), SigmoidTransform()]
    response_dist = dist.TransformedDistribution(base_dist, transforms)
    return pyro.sample("y", response_dist)


def iqr(array):
    return np.percentile(array, 75) - np.percentile(array, 25)


def main(num_experiments, num_runs, plot=True):

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
            ba_kwargs = {"num_samples": 100, "num_steps": 500, "guide": my_guide,
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
                mu_dict, scale_tril_dict = my_guide.get_params({"y": y}, AB_test_reff_6d_10n_12p, ["w1"])
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

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        x = np.array([0., 1.])
        plt.bar(x, height=means, yerr=iqrs, tick_label=['OED', 'Random design'],
                color=plt.rcParams['axes.prop_cycle'].by_key()['color'][0:2],
                capsize=10)
        plt.ylabel("Final posterior entropy")
        plt.show()

    else:
        print("Mean posterior entropy", means)
        print("IQR posterior entropy", iqrs)


if __name__ == "__main__":
    assert pyro.__version__.startswith('0.3.0')
    parser = argparse.ArgumentParser(description="Sigmoid iterated experiment design")
    parser.add_argument("--num-experiments", nargs="?", default=5, type=int)
    parser.add_argument("--num-runs", nargs="?", default=5, type=int)
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--plot', dest='plot', action='store_true')
    feature_parser.add_argument('--no-plot', dest='plot', action='store_false')
    parser.set_defaults(feature=True)
    args = parser.parse_args()
    main(args.num_experiments, args.num_runs, args.plot)
