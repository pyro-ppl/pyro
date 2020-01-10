# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import json
import uuid
import functools

import torch

import pyro
import pyro.poutine as poutine
from pyro.infer import TraceEnum_ELBO

from model import model_generic, guide_generic
from seal_data import prepare_seal


def aic_num_parameters(model, guide=None):
    """
    hacky AIC param count that includes all parameters in the model and guide
    """

    def _size(tensor):
        """product of shape"""
        s = 1
        for d in tensor.shape:
            s = s * d
        return s

    with poutine.block(), poutine.trace(param_only=True) as param_capture:
        TraceEnum_ELBO(max_plate_nesting=2).differentiable_loss(model, guide)

    return sum(_size(node["value"]) for node in param_capture.trace.nodes.values())


def run_expt(args):

    data_dir = args["folder"]
    dataset = "seal"  # args["dataset"]
    seed = args["seed"]
    optim = args["optim"]
    lr = args["learnrate"]
    timesteps = args["timesteps"]
    schedule = [] if not args["schedule"] else [int(i) for i in args["schedule"].split(",")]
    random_effects = {"group": args["group"], "individual": args["individual"]}

    pyro.enable_validation(args["validation"])
    pyro.set_rng_seed(seed)  # reproducible random effect parameter init

    filename = os.path.join(data_dir, "prep_seal_data.csv")
    config = prepare_seal(filename, random_effects)

    model = functools.partial(model_generic, config)  # for JITing
    guide = functools.partial(guide_generic, config)

    # count the number of parameters once
    num_parameters = aic_num_parameters(model, guide)

    losses = []
    # SGD
    if optim == "sgd":
        loss_fn = TraceEnum_ELBO(max_plate_nesting=2).differentiable_loss
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss_fn(model, guide)
        params = [site["value"].unconstrained() for site in param_capture.trace.nodes.values()]
        optimizer = torch.optim.Adam(params, lr=lr)

        if schedule:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=schedule, gamma=0.5)
            schedule_step_loss = False
        else:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            schedule_step_loss = True

        for t in range(timesteps):

            optimizer.zero_grad()
            loss = loss_fn(model, guide)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item() if schedule_step_loss else t)
            losses.append(loss.item())

            print("Loss: {}, AIC[{}]: ".format(loss.item(), t),
                  2. * loss + 2. * num_parameters)

    # LBFGS
    elif optim == "lbfgs":
        loss_fn = TraceEnum_ELBO(max_plate_nesting=2).differentiable_loss
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss_fn(model, guide)
        params = [site["value"].unconstrained() for site in param_capture.trace.nodes.values()]
        optimizer = torch.optim.LBFGS(params, lr=lr)

        if schedule:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=schedule, gamma=0.5)
            schedule_step_loss = False
        else:
            schedule_step_loss = True
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        for t in range(timesteps):
            def closure():
                optimizer.zero_grad()
                loss = loss_fn(model, guide)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            scheduler.step(loss.item() if schedule_step_loss else t)
            losses.append(loss.item())
            print("Loss: {}, AIC[{}]: ".format(loss.item(), t),
                  2. * loss + 2. * num_parameters)

    else:
        raise ValueError("{} not supported optimizer".format(optim))

    aic_final = 2. * losses[-1] + 2. * num_parameters
    print("AIC final: {}".format(aic_final))

    results = {}
    results["args"] = args
    results["sizes"] = config["sizes"]
    results["likelihoods"] = losses
    results["likelihood_final"] = losses[-1]
    results["aic_final"] = aic_final
    results["aic_num_parameters"] = num_parameters

    if args["resultsdir"] is not None and os.path.exists(args["resultsdir"]):
        re_str = "g" + ("n" if args["group"] is None else "d" if args["group"] == "discrete" else "c")
        re_str += "i" + ("n" if args["individual"] is None else "d" if args["individual"] == "discrete" else "c")
        results_filename = "expt_{}_{}_{}.json".format(dataset, re_str, str(uuid.uuid4().hex)[0:5])
        with open(os.path.join(args["resultsdir"], results_filename), "w") as f:
            json.dump(results, f)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--group", default="none", type=str)
    parser.add_argument("-i", "--individual", default="none", type=str)
    parser.add_argument("-f", "--folder", default="./", type=str)
    parser.add_argument("-o", "--optim", default="sgd", type=str)
    parser.add_argument("-lr", "--learnrate", default=0.05, type=float)
    parser.add_argument("-t", "--timesteps", default=1000, type=int)
    parser.add_argument("-r", "--resultsdir", default="./results", type=str)
    parser.add_argument("-s", "--seed", default=101, type=int)
    parser.add_argument("--schedule", default="", type=str)
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    if args.group == "none":
        args.group = None
    if args.individual == "none":
        args.individual = None

    run_expt(vars(args))
