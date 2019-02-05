from __future__ import absolute_import, division, print_function

import logging
import os

import numpy as np
import pandas as pd
import pyreadr

import torch

import pyro
import pyro.distributions as dist


def prepare_seal(filename, random_effects):
    seal_df = pyreadr.read_r(filename)['hsData']
    obs_keys = ["step", "angle", "omega"]
    # data format for z1, z2:
    # single tensor with shape (individual, group, time, coords)
    observations = torch.zeros((20, 2, 1800, len(obs_keys))).fill_(float("-inf"))
    for g, (group, group_df) in enumerate(seal_df.groupby("sex")):
        for i, (ind, ind_df) in enumerate(group_df.groupby("ID")):
            for o, obs_key in enumerate(obs_keys):
                observations[i, g, 0:len(ind_df), o] = torch.tensor(ind_df[obs_key].values)

    observations[torch.isnan(observations)] = float("-inf")

    # make masks
    # mask_i should mask out individuals, it applies at all timesteps
    mask_i = (observations > float("-inf")).any(dim=-1).any(dim=-1)  # time nonempty

    # mask_t handles padding for time series of different length
    mask_t = (observations > float("-inf")).all(dim=-1)   # include non-inf

    # temporary hack to avoid zero-inflation issues
    # observations[observations == 0.] = 1e-4
    observations[(observations == 0.) | (observations == float("-inf"))] = 1e-4
    assert not torch.isnan(observations).any()

    # observations = observations[..., 5:11, :]  # truncate for testing

    config = {
        "sizes": {
            "state": 3,
            "random": 4,
            "group": observations.shape[1],
            "individual": observations.shape[0],
            "timesteps": observations.shape[2],
        },
        "group": {"random": random_effects["group"], "fixed": None},
        "individual": {"random": random_effects["individual"], "fixed": None, "mask": mask_i},
        "timestep": {"random": None, "fixed": None, "mask": mask_t},
        "observations": {
            "step": observations[..., 0],
            "angle": observations[..., 1],
            "omega": observations[..., 2],
        },
    }

    return config
