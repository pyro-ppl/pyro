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

    for level in ["group", "individual",]:  #  "timestep"]:
        # count random effect parameters
        if config[level]["random"] == "discrete":
            num_params += _size(pyro.param("probs_e_{}".format(level)))
            num_params += _size(pyro.param("theta_{}".format(level)))
        elif config[level]["random"] == "continuous":
            num_params += _size(pyro.param("loc_{}".format(level)))
            num_params += _size(pyro.param("scale_{}".format(level)))

        # count fixed effect parameters
        if config[level]["fixed"]:
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


def run_expt(data_dir, dataset, random_effects, seed):

    pyro.set_rng_seed(seed)  # reproducible random effect parameter init

    if dataset == "seal":
        filename = os.path.join(data_dir, "prep_seal_data.RData")
        config = prepare_seal(filename, random_effects)
    elif dataset == "shark":
        filename = os.path.join(data_dir, "gws_full.xlsx")
        config = prepare_shark(filename, random_effects)

    model = lambda: model_generic(config)  # for JITing
    guide = lambda: guide_generic(config)
    svi = pyro.infer.SVI(model, guide, loss=TraceEnum_ELBO(max_plate_nesting=2), optim=pyro.optim.Adam({"lr": 0.05}))
    for _ in range(1000):
        print(svi.step())
    print("AIC: {}".format(aic(model, guide, config)))


if __name__ == "__main__":

    pyro.enable_validation(False)

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--seed", default=101, type=int)
    parser.add_argument("-d", "--dataset", default="seal", type=str)
    parser.add_argument("-g", "--group", default=None, type=str)
    parser.add_argument("-i", "--individual", default=None, type=str)
    parser.add_argument("-f", "--folder", default="/home/eli/wsl/momentuHMM/vignettes/", type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()

    data_dir = args.folder
    dataset = args.dataset
    seed = args.seed
    random_effects = {"group": args.group, "individual": args.individual}

    with pyro.util.ignore_jit_warnings():
        run_expt(data_dir, dataset, random_effects, seed)
