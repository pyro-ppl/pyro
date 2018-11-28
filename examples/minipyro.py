from __future__ import absolute_import, division, print_function

import argparse

import torch

import pyro.distributions as dist


def main(args):
    if args.full_pyro:
        import pyro
        from pyro.infer import SVI, Trace_ELBO
        from pyro.optim import Adam
        elbo = Trace_ELBO()
    else:
        import pyro.contrib.minipyro as pyro
        from pyro.contrib.minipyro import SVI, Adam, elbo

    def model(data):
        loc = pyro.sample("loc", dist.Normal(0., 1.))
        pyro.sample("obs", dist.Normal(loc, 1.), obs=data)

    def guide(data):
        loc_loc = pyro.param("loc_loc", torch.tensor(0.))
        loc_scale = pyro.param("loc_scale_log", torch.tensor(0.)).exp()
        pyro.sample("loc", dist.Normal(loc_loc, loc_scale))

    torch.manual_seed(0)
    data = torch.randn(100) + 3.0
    pyro.get_param_store().clear()

    svi = SVI(model, guide, Adam({"lr": args.learning_rate}), elbo)
    for step in range(args.num_steps):
        loss = svi.step(data)
        if step % 100 == 0:
            print("step {} loss = {}".format(step, loss))

    for name, value in pyro.get_param_store().items():
        print("{} = {}".format(name, value.detach().cpu().numpy()))

    assert (pyro.param("loc_loc") - 3.0).abs() < 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mini Pyro demo")
    parser.add_argument("-f", "--full-pyro", action="store_true", default=False)
    parser.add_argument("-n", "--num-steps", default=1001, type=int)
    parser.add_argument("-lr", "--learning-rate", default=0.02, type=float)
    args = parser.parse_args()
    main(args)
