from __future__ import absolute_import, division, print_function

import argparse
import logging
import os

import numpy as np
import torch

import pyro
from pyro.contrib.autoguide import AutoDelta
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO
from pyro.optim import Adam
from pyro.util import ignore_jit_warnings

from model import model_generic, guide_generic
from seal_data import prepare_seal
from shark_data import prepare_shark


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


def run_expt(data_dir, dataset, random_effects, seed, optim):

    pyro.set_rng_seed(seed)  # reproducible random effect parameter init

    if dataset == "seal":
        filename = os.path.join(data_dir, "prep_seal_data.RData")
        config = prepare_seal(filename, random_effects)
    elif dataset == "shark":
        filename = os.path.join(data_dir, "gws_full.xlsx")
        config = prepare_shark(filename, random_effects)

    model = lambda: model_generic(config)  # for JITing
    guide = lambda: guide_generic(config)

    # SGD
    if optim == "sgd":
        loss_fn = TraceEnum_ELBO(max_plate_nesting=2).differentiable_loss
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss_fn(model, guide)
        params = [site["value"].unconstrained() for site in param_capture.trace.nodes.values()]
        optimizer = torch.optim.Adam(params, lr=0.05)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        for t in range(1000):

            optimizer.zero_grad()
            loss = loss_fn(model, guide)
            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            print("Loss: {}, AIC[{}]: ".format(loss, t), 
                  2. * loss * 2. + aic_num_parameters(config))

    # LBFGS
    elif optim == "lbfgs":
        loss_fn = TraceEnum_ELBO(max_plate_nesting=2).differentiable_loss
        with pyro.poutine.trace(param_only=True) as param_capture:
            loss_fn(model, guide)
        params = [site["value"].unconstrained() for site in param_capture.trace.nodes.values()]
        optimizer = torch.optim.LBFGS(params, lr=0.05)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        for t in range(100):
            def closure():
                optimizer.zero_grad()
                loss = loss_fn(model, guide)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            scheduler.step(loss.item())
            print("Loss: {}, AIC[{}]: ".format(loss, t), 
                  2. * loss * 2. + aic_num_parameters(config))

    else:
        raise ValueError("{} not supported optimizer".format(optim))

    aic_final = aic(model, guide, config)
    print("AIC final: {}".format(aic_final))
    return aic_final


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=101, type=int)
    parser.add_argument("-d", "--dataset", default="seal", type=str)
    parser.add_argument("-g", "--group", default=None, type=str)
    parser.add_argument("-i", "--individual", default=None, type=str)
    parser.add_argument("-f", "--folder", default="/home/eli/wsl/momentuHMM/vignettes/", type=str)
    parser.add_argument("-o", "--optim", default="sgd", type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    parser.add_argument('--validation', action='store_true')
    args = parser.parse_args()

    pyro.enable_validation(args.validation)

    data_dir = args.folder
    dataset = args.dataset
    seed = args.seed
    optim = args.optim
    random_effects = {"group": args.group, "individual": args.individual}

    run_expt(data_dir, dataset, random_effects, seed, optim)
