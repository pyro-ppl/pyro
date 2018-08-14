import argparse
import torch
from torch.nn.functional import softplus

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import vi_ape

from models.bayes_linear import two_group_bernoulli


model, guide = two_group_bernoulli(torch.tensor([1.]), torch.tensor([.5]))


def spherical_design_tensor(d):
    return torch.stack([torch.cos(d), -torch.sin(d)], -1)


def main(num_vi_steps):

    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    def estimated_ape(designs, ydist):
        design_tensor = spherical_design_tensor(designs)
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

    X = torch.tensor([[0., 0.], [0., 1.5]])
    y = estimated_ape(X, dist.Bernoulli(torch.tensor([0.5])))
    print(y)
    # pyro.clear_param_store()
    # gpmodel = gp.models.GPRegression(
    #     X, y, gp.kernels.Matern52(input_dim=1, lengthscale=torch.tensor(5.)),
    #     noise=torch.tensor(0.1), jitter=1e-6)
    # gpmodel.optimize()
    # gpbo = GPBayesOptimizer(estimated_ape, constraints.interval(0, 6.29), gpmodel)
    # print(gpbo.run(num_steps=num_bo_steps, num_acquisitions=num_acquisitions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Item response experiment design using VI")
    parser.add_argument("-n", "--num-vi-steps", nargs="?", default=5000, type=int)
    # parser.add_argument('--num-acquisitions', nargs="?", default=10, type=int)
    # parser.add_argument('--num-bo-steps', nargs="?", default=6, type=int)
    args = parser.parse_args()
    main(args.num_vi_steps)
