from __future__ import absolute_import, division, print_function

import argparse
import os
import itertools
import collections
import subprocess
import multiprocessing

from experiment import run_expt


config_components = collections.OrderedDict(
    dataset = ["seal",],  #  "seal",],
    # dataset = ["shark", "seal"],
    group = ["none", "discrete", "continuous"],
    individual = ["none", "discrete", "continuous"],
    folder = ["./"],
    optim = ["sgd",],
    learnrate = [0.1, 0.05, 0.01, 0.005],
    timesteps = [300, 600,],
    resultsdir = ["./results",],
    seed = [101, 102, 103, 104, 105],
    schedule = ["", "10,20,60", "50,100,400", "100,200,600"],
    validation = [False,],
)

configs = [{k: v for k, v in zip(config_components.keys(), c)}
           for c in itertools.product(*list(config_components.values()))]

with multiprocessing.Pool(24) as p:
    p.map(run_expt, configs)
