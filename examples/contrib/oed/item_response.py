import argparse
import torch
from torch.nn.functional import softplus

import pyro
import pyro.distributions as dist
from pyro import optim
from pyro.infer import TraceEnum_ELBO
from pyro.contrib.oed.eig import vi_ape

# Number of layers of hierarchy (1 layer- classical linear model)
layers = 2
# Parameters controlling the prior variance
A = 20.
B = 10.
beta = torch.tensor(1.)


def model(design_tensor, participant_id):
    """Hierarchical logistic regression model.

    :param torch.tensor design_tensor: a `batch_dims x n x p`
        tensor giving the features of the `n` responses (e.g.
        the features of the `n` objects to be given at this time)
    :param list participant_id: a hierarchical list of the groups
        that the current participant is assigned to. The first element
        indexes the highest level group. For instance, a geographical
        hierarchy list could look like
        `["United States", "California", "Santa Rosa"]`
    """
    # batch x n
    response_shape = list(design_tensor.shape)[:-1]
    # batch x 1 x p
    coef_shape = list(design_tensor.shape)
    coef_shape[-2] = 1

    intercept = torch.zeros(response_shape)
    coef = torch.zeros(coef_shape)

    for layer in range(layers):
        layer_name = "_".join([str(layer)] + participant_id[:layer])
        # Prior sds decay exponentially with layer
        intercept_sd = torch.ones(response_shape)*A*torch.exp(-beta*layer)
        intercept_dist = dist.Normal(0., intercept_sd).independent(1)
        intercept += pyro.sample(layer_name + "_intercept", intercept_dist)

        coef_sd = torch.ones(coef_shape)*B*torch.exp(-beta*layer)
        coef_dist = dist.Normal(0., coef_sd).independent(2)
        coef += pyro.sample(layer_name + "_coef", coef_dist)

    logit_p = torch.matmul(design_tensor, coef.t()).squeeze(-1) + intercept
    # Binary outcomes - responses to `n` items shown to given participant
    return pyro.sample("y", dist.Bernoulli(logits=logit_p).independent(1))


def guide(design_tensor, participant_id):
    # batch x n
    response_shape = list(design_tensor.shape)[:-1]
    # batch x 1 x p
    coef_shape = list(design_tensor.shape)
    coef_shape[-2] = 1
    # define our variational parameters, sample mean-field
    for layer in range(layers):
        # Local intercept
        layer_name = "_".join([str(layer)] + participant_id[:layer])
        intercept_mean = pyro.param(layer_name + "_intercept_mean",
                                    torch.zeros(response_shape))
        intercept_sd = softplus(pyro.param(layer_name + "_intercept_sd",
                                10.*torch.ones(response_shape)))
        intercept_dist = dist.Normal(intercept_mean, intercept_sd).independent(1)
        pyro.sample(layer_name + "_intercept", intercept_dist)

        # Local coefficient
        coef_mean = pyro.param(layer_name + "_coef_mean",
                               torch.zeros(coef_shape))
        coef_sd = softplus(pyro.param(layer_name + "_coef_sd",
                           10.*torch.ones(coef_shape)))
        coef_dist = dist.Normal(coef_mean, coef_sd).independent(2)
        pyro.sample(layer_name + "_coef", coef_dist)


def spherical_design_tensor(d):
    return torch.stack([torch.cos(d), -torch.sin(d)], -1)


def main(num_vi_steps):

    pyro.set_rng_seed(42)
    pyro.clear_param_store()

    def estimated_ape(designs, participant, ydist):
        design_tensor = spherical_design_tensor(designs)
        est_ape = vi_ape(
            lambda d: model(d, participant),
            design_tensor,
            observation_labels="y",
            vi_parameters={
                "guide": lambda d: guide(d, participant),
                "optim": optim.Adam({"lr": 0.0025}),
                "loss": TraceEnum_ELBO(strict_enumeration_warning=False).differentiable_loss,
                "num_steps": num_vi_steps},
            is_parameters={"num_samples": 10},
            y_dist=ydist
        )
        return est_ape

    participant = ["a"]
    X = torch.tensor([[0., 0.], [0., 1.5]])
    y = estimated_ape(X, participant, None)
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
