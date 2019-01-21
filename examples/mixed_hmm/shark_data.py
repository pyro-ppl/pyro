from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import pandas as pd
import pyreadr

import torch

import pyro
import pyro.distributions as dist


def _encode_shark_df(tracks_df, summary_df):
    """
    This function converts the excel-derived encoding of the original data
    to something more uniform and machine-readable for preparing experiments
    """
    shark_df = tracks_df.copy()

    # split up the columns
    # group column
    shark_df["sex"] = shark_df["Shark sex and track number"].str[2:3]
    # individual column (track)
    shark_df["ID"] = shark_df["Shark sex and track number"]

    # animal column
    shark_df["id_name"] = shark_df["Shark sex and track number"].str[0:5]
    shark_df["id_name"] = shark_df["id_name"].str.strip("T ")

    # covariates
    # tail length column from summary
    shark_df.loc[:, "TL"] = pd.Series(np.random.randn(len(shark_df["Latitude"])), index=shark_df.index)
    for individual_name in shark_df["id_name"].unique():
        individual_length = summary_df[summary_df["Shark ID"] == individual_name]["TL (cm)"].values[0]
        shark_df["TL"][shark_df["id_name"] == individual_name] = individual_length

    # make chum value into something usable
    shark_df.loc[:, "chum"] = pd.Series(np.random.randn(len(shark_df["Latitude"])),
                                        index=shark_df.index)
    shark_df["chum"][shark_df["Cage Dive Boat"] == 'x'] = 1.
    shark_df["chum"][shark_df["Cage Dive Boat"] != 'x'] = 0.

    # time covariates
    # XXX gross pandas stuff
    shark_df.loc[:, "time_num"] = pd.Series(np.array([t.hour + t.minute / 60. for t in list(shark_df["Time"].values)], dtype=np.float32), index=shark_df.index)
    shark_df["sin_time"] = shark_df["time_num"].apply(lambda t: np.sin(t * np.pi * 2. / 288.))
    shark_df["cos_time"] = shark_df["time_num"].apply(lambda t: np.cos(t * np.pi * 2. / 288.))

    # convert lat/lon to step/angle
    # 1. convert to x/y projection
    # 2. compute length of each difference
    # 3. compute angle between differences
    shark_df["step"] = compute_step()
    shark_df["angle"] = compute_angle()

    return shark_df


def prepare_shark(filename, random_effects):
    tracks_df = pd.read_excel(filename, sheet_name=0)
    summary_df = pd.read_excel(filename, sheet_name=1)

    shark_df = _encode_shark_df(tracks_df, summary_df)
    obs_keys = ["Latitude", "Longitude"]
    # obs_keys = ["step", "angle"]  # TODO

    # data format for z1, z2:
    # single tensor with shape (individual, group, time, coords)
    observations = torch.zeros((100, 2, 270, len(obs_keys))).fill_(float("-inf"))
    for g, (group, group_df) in enumerate(shark_df.groupby("sex")):
        for i, (ind, ind_df) in enumerate(group_df.groupby("ID")):
            for o, obs_key in enumerate(obs_keys):
                observations[i, g, 0:len(ind_df), o] = torch.tensor(ind_df[obs_key].values)

    # make covariates
    observations = torch.zeros((100, 2, 270, len(obs_keys))).fill_(float("-inf"))
    for g, (group, group_df) in enumerate(shark_df.groupby("sex")):
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
            "random": 10,
            "group": observations.shape[1],
            "individual": observations.shape[0],
            "timesteps": observations.shape[2],
        },
        "group": {"random": random_effects["group"], "fixed": None},
        "individual": {"random": random_effects["individual"], "fixed": None, "mask": mask_i},
        "timestep": {"random": None, "fixed": None, "mask": mask_t},
        "observations": {
            "Latitude": {"dist": dist.Normal, "zi": False, "values": observations[..., 0]},
            "Longitude": {"dist": dist.Normal, "zi": False, "values": observations[..., 1]},
        },
    }

    return config

