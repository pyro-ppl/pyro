# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0
# flake8: noqa

import pickle
import numpy as np
import pandas as pd

import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import r as reval
from rpy2.robjects.conversion import localconverter, rpy2py


def load_data(epi_path, phy_path, dt):
    base = importr("base")
    ape = importr("ape")
    lubridate = importr("lubridate")
    epigenr = importr("EpiGenR")

    rpy2.robjects.r('''
epi_path <- "~/Downloads/california_timeseries.txt"
phy_path <- "~/Downloads/california_tree.nwk"
dt = (1/4)/365

epi_data <- read.table(epi_path)
epi_data <- epi_data[2:length(epi_data[, 1]), ]
epi_data[, 1] <- lubridate::decimal_date(as.Date(epi_data[, 1], format="%Y-%m-%d"))

phy_raw_data <- read.tree(phy_path)
phy_tree_data <- coalescent.intervals.datedPhylo(reorder.phylo(phy_raw_data, "postorder"))
phy_data <- coal.intervals.in.discrete.time(phy_tree_data, dt)

epi_gen_data <- align_epi_gen_data(epi_data, phy_data, dt, get_last_tip_time(phy_tree_data))
    ''')

    epi_gen_data = rpy2.robjects.r['epi_gen_data']

    py_epi_gen_data = rpy2py(epi_gen_data)
    with localconverter(rpy2.robjects.default_converter + pandas2ri.converter):
        py_epi_data = rpy2py(reval('epi_gen_data$epi'))

    py_epi_data = np.array(py_epi_data['V2'], dtype=np.float32)

    r_gen_data = rpy2py(reval('epi_gen_data$gen'))
    py_gen_data = [
        {'binomial': np.array(r_gen_data[i][0]), 'intervals': np.array(r_gen_data[i][1])}
        for i in range(len(r_gen_data))
    ]
    return {'epi': py_epi_data, 'phy': py_gen_data}


if __name__ == "__main__":
    epi_path = "/home/eli/Downloads/california_timeseries.txt"
    phy_path = "/home/eli/Downloads/california_tree.nwk"
    dt = (1/4.)/365.
    data = load_data(epi_path, phy_path, dt)
    with open("/home/eli/processed_epigen_data.pkl", 'wb') as f:
        pickle.dump(data, f)
