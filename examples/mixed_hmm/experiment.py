from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import json
import uuid

import numpy as np
import torch

import pyro
from pyro.infer import SVI, TraceEnum_ELBO

from model import model_generic, guide_generic
from seal_data import prepare_seal


def aic_num_parameters(config):

    def _size(tensor):
        """product of shape"""
        s = 1
        for d in tensor.shape:
            s = s * d
        return s

    num_params = 0

    for level in ["group", "individual", "timestep"]:
        # count random effect parameters
        if config[level]["random"] == "discrete":
            num_params += _size(pyro.param("probs_e_{}".format(level)))
            num_params += _size(pyro.param("theta_{}".format(level)))
        elif config[level]["random"] == "continuous":
            num_params += _size(pyro.param("loc_{}".format(level)))
            num_params += _size(pyro.param("scale_{}".format(level)))

        # count fixed effect parameters
        if config[level]["fixed"] is not None:
            num_params += _size(pyro.param("beta_{}".format(level)))

    # count likelihood parameters
    for coord, coord_config in config["observations"].items():
        num_params += sum([
            _size(pyro.param("{}_param_{}".format(coord, arg_name)))
            for arg_name in coord_config["dist"].arg_constraints.keys()
        ])
        # count zero-inflation parameters
        if coord_config["zi"]:
            num_params += _size(pyro.param("{}_zi_param".format(coord)))

    return num_params


def aic(model, guide, config):
    neg_log_likelihood = TraceEnum_ELBO(max_plate_nesting=2).differentiable_loss(model, guide)
    num_params = aic_num_parameters(config)
    return 2. * neg_log_likelihood + 2. * num_params


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

    model = lambda: model_generic(config)  # for JITing
    guide = lambda: guide_generic(config)

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
                  2. * loss + 2. * aic_num_parameters(config))

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
                  2. * loss + 2. * aic_num_parameters(config))

    else:
        raise ValueError("{} not supported optimizer".format(optim))

    aic_final = aic(model, guide, config)
    print("AIC final: {}".format(aic_final))

    results = {}
    results["args"] = args
    results["sizes"] = config["sizes"]
    results["likelihoods"] = losses
    results["likelihood_final"] = losses[-1]
    results["aic_final"] = aic_final.item()
    results["aic_num_parameters"] = aic_num_parameters(config)

    if args["resultsdir"] is not None:
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
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    if args.group == "none":
        args.group = None
    if args.individual == "none":
        args.individual = None

    run_expt(vars(args))
