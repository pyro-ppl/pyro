from __future__ import absolute_import, division, print_function

import argparse
import torch

import pyro
from pyro.contrib.oed.eig import naive_rainforth_eig
from pyro.contrib.glmm import lmer_model

"""
Item response example.

Items are characterized by attributes living on the unit circle S^1.
Binary outcomes are combination of individual and global effects, according
to the logistic regression model

    :math:`logit(p)=Xw_{global} + Xw_{individual}`

In this example, we compare three designs, aiming to gain information about
global and individual effects.

In the current implementation, the covariance matrix for the random effects
is assumed to be a known diagonal matrix. The estimation problem is then to
learn the actual random effect coefficients.
"""

# First parameter grouping represents global effects
# Second parameter grouping represent individual effects
# In this example, there are two individuals
# Choose inverse gamma distribution to have mean ~20, variance ~1
model = lmer_model(torch.tensor([10., 10.]), 2, torch.tensor([400., 400.]),
                   torch.tensor([8000., 8000.]), response="bernoulli")


def build_design_tensor(item_thetas, individual_assignment):
    """
    Given a sequence of angles representing objects on S^1, and
    assignments of those objects to individuals, creates a design tensor
    of the appropriate dimensions.

    :param torch.Tensor item_thetas: a tensor of dimension batch x n representing
        the item to be used at the nth trial
    :param torch.Tensor individual_assignment: a tensor of dimension batch x n x 2
        with 0-1 rows indicating the individual to assign item n to
    """
    # batch x n x 2
    item_features = torch.stack([item_thetas.cos(), -item_thetas.sin()], dim=-1)
    ind1 = individual_assignment[..., 0].unsqueeze(-1)*item_features
    ind2 = individual_assignment[..., 1].unsqueeze(-1)*item_features
    # batch x n x 6
    return torch.cat([item_features, ind1, ind2], dim=-1)


def main(N, M):

    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    # Assignment items to two different individuals
    individual_assignment = torch.tensor([[1., 0.], [0., 1.]])
    # Design 1: the same item to both people
    # Design 2-3: different items to different people
    item_thetas = torch.tensor([[0., 0.], [0., .5], [0., 1.]])
    design_tensor = build_design_tensor(item_thetas, individual_assignment)
    print("Design tensor", design_tensor)
    y = naive_rainforth_eig(model, design_tensor, observation_labels="y",
                            target_labels=["w", "u", "G_u"], N=N, M=M)
    print("EIG", y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Item response experiment design using NMC")
    parser.add_argument("-N", nargs="?", default=5000, type=int)
    parser.add_argument("-M", nargs="?", default=5000, type=int)
    args = parser.parse_args()
    main(args.N, args.M)
