import argparse
import torch

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import vi_ape

from models.bayes_linear import two_group_bernoulli

"""
Item response example.

Items are characterized by attributes living on the unit circle S^1.
Binary outcomes are combination of individual and global effects.

TODO: Compare with estimation using Rainforth estimators and DV
"""

# First parameter grouping represents global effects
# Second parameter grouping represent individual effects
# In this example, there are two individuals
# Individual-level effects have lower sd (more tightly clustered to 0)
model, guide = two_group_bernoulli(torch.tensor([1., 1.]), torch.tensor([.5, .5, .5, .5]))


def build_design_tensor(item_thetas, individual_assignment):
    """
    `item_thetas` should be a tensor of dimension batch x n representing
    the item to be used at the nth trial
    `individual_assignment` should be a tensor of dimension batch x n x 2
    with 0-1 rows indicating the individual to assign item n to
    """
    # batch x n x 2
    item_features = torch.stack([item_thetas.cos(), -item_thetas.sin()], dim=-1)
    ind1 = individual_assignment[..., 0]*item_features
    ind2 = individual_assignment[..., 1]*item_features
    return torch.cat([item_features, ind1, ind2], dim=-1)


def main(num_vi_steps):

    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    def estimated_ape(design_tensor, ydist):
        est_ape = vi_ape(
            model,
            design_tensor,
            observation_labels="y",
            vi_parameters={
                "guide": guide,
                "optim": optim.Adam({"lr": 0.0025}),
                "loss": TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss,
                "num_steps": num_vi_steps},
            is_parameters={"num_samples": 10},
            y_dist=ydist
        )
        return est_ape

    # Assignment items to two different individuals
    individual_assignment = torch.tensor([[1., 0.], [0., 1.]])
    # Design 1: the same item to both people
    # Design 2: different items to different people
    item_thetas = torch.tensor([[0., 0.], [0., 1.5]])
    design_tensor = build_design_tensor(item_thetas, individual_assignment)
    y = estimated_ape(design_tensor, dist.Bernoulli(torch.tensor([0.5, 0.5])))
    print(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Item response experiment design using VI")
    parser.add_argument("-n", "--num-vi-steps", nargs="?", default=5000, type=int)
    args = parser.parse_args()
    main(args.num_vi_steps)
