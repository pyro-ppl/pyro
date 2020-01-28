# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from urllib.request import urlopen

import pandas as pd

import torch


MISSING = 1e-6


def download_seal_data(filename):
    """download the preprocessed seal data and save it to filename"""
    url = "https://d2hg8soec8ck9v.cloudfront.net/datasets/prep_seal_data.csv"
    with open(filename, "wb") as f:
        f.write(urlopen(url).read())


def prepare_seal(filename, random_effects):

    if not os.path.exists(filename):
        download_seal_data(filename)

    seal_df = pd.read_csv(filename)
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
    # observations[observations == 0.] = MISSING
    observations[(observations == 0.) | (observations == float("-inf"))] = MISSING
    assert not torch.isnan(observations).any()

    # observations = observations[..., 5:11, :]  # truncate for testing

    config = {
        "MISSING": MISSING,
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
