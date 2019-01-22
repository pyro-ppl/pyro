from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import pandas as pd
import pyreadr

import utm

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

    # add back conversion of lat/lon to step/angle
    # 1. convert to x/y projection with utm
    # 2. compute length of each difference
    # 3. compute angle between differences
    shark_df["step"] = pd.Series(np.zeros((len(shark_df),), dtype=np.float32),
                                 index=shark_df.index)
    shark_df["angle"] = pd.Series(np.zeros((len(shark_df),), dtype=np.float32),
                                  index=shark_df.index)
    for trackname, track_df in shark_df.groupby("ID"):
        track_lat, track_lon = track_df["Latitude"], track_df["Longitude"]
        x, y = np.zeros((len(track_lat),)), np.zeros((len(track_lon),))
        for i, (lat, lon) in enumerate(zip(list(track_lat.values), list(track_lon.values))):
            x[i], y[i], _, _, = utm.from_latlon(lat, lon)

        xy = np.stack([x, y], axis=-1) / 1000.  # km
        step = xy[1:] - xy[:-1]
        step_length = np.sqrt(np.einsum("ab,ab->a", step, step))
        dstep = step[1:] - step[:-1]
        step_angle = np.arccos(
            np.einsum("ab,ab->a", step[1:], dstep) / (step_length[1:] * np.sqrt(np.einsum("ab,ab->a", dstep, dstep))))

        step_length[np.isnan(step_length)] = 0.
        step_angle[np.isnan(step_angle)] = 0.

        if len(track_df) > 2:  # cover a weird edge case "WSF6 T4 (T5)"
            shark_df["step"][shark_df.ID == trackname] = np.concatenate([np.zeros((1,), dtype=np.float32), step_length])
            shark_df["angle"][shark_df.ID == trackname] = np.concatenate([np.zeros((2,), dtype=np.float32), step_angle])

            # sanity checks
            assert (shark_df["step"][shark_df.ID == trackname].values != 0.).any()
            assert (shark_df["angle"][shark_df.ID == trackname].values != 0.).any()

    return shark_df


def prepare_shark(filename, random_effects):

    tracks_df = pd.read_excel(filename, sheet_name=0)
    summary_df = pd.read_excel(filename, sheet_name=1)

    shark_df = _encode_shark_df(tracks_df, summary_df)
    obs_keys = ["step", "angle"]

    # data format for z1, z2:
    # single tensor with shape (individual, group, time, coords)
    observations = torch.zeros((100, 2, 270, len(obs_keys))).fill_(float("-inf"))
    for g, (group, group_df) in enumerate(shark_df.groupby("sex")):
        for i, (ind, ind_df) in enumerate(group_df.groupby("ID")):
            for o, obs_key in enumerate(obs_keys):
                observations[i, g, 0:len(ind_df), o] = torch.tensor(ind_df[obs_key].values)

    # make covariates: chum (timestep), time sin/cos (timestep), size (individual)
    individual_cov = torch.zeros((100, 2, 1)).fill_(float("-inf"))
    timestep_cov = torch.zeros((100, 2, 270, 4)).fill_(float("-inf"))
    for g, (group, group_df) in enumerate(shark_df.groupby("sex")):
        for i, (ind, ind_df) in enumerate(group_df.groupby("ID")):
            individual_cov[i, g, 0:1] = torch.tensor(ind_df["TL"].values[0:1])
            timestep_cov[i, g, 0:len(ind_df), 0] = torch.tensor(ind_df["sin_time"].values)
            timestep_cov[i, g, 0:len(ind_df), 1] = torch.tensor(ind_df["cos_time"].values)
            # chum is an indicator so we expand as one-hot
            timestep_cov[i, g, 0:len(ind_df), 2] = torch.tensor(ind_df["chum"].values)
            timestep_cov[i, g, 0:len(ind_df), 3] = torch.tensor(1. - ind_df["chum"].values)

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

    timestep_cov[(timestep_cov == 0.) | (timestep_cov == float("-inf"))] = 1e-4
    individual_cov[(individual_cov == 0.) | (individual_cov == float("-inf"))] = 1e-4

    # observations = observations[..., 5:11, :]  # truncate for testing

    config = {
        "sizes": {
            "state": 2,
            "random": 3,
            "group": observations.shape[1],
            "individual": observations.shape[0],
            "timesteps": observations.shape[2],
        },
        "group": {"random": random_effects["group"], "fixed": None},
        "individual": {"random": random_effects["individual"], "fixed": individual_cov, "mask": mask_i},
        "timestep": {"random": None, "fixed": timestep_cov, "mask": mask_t},
        "observations": {
            "step": {"dist": dist.Gamma, "zi": True, "values": observations[..., 0]},
            "angle": {"dist": dist.VonMises, "zi": False, "values": observations[..., 1]},
        },
    }

    return config
