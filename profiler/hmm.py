# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import pickle
import re
import subprocess
import sys
from collections import defaultdict
from os.path import join, abspath

from numpy import median

from pyro.util import timed


EXAMPLES_DIR = join(abspath(__file__), os.pardir, os.pardir, "examples")


def main(args):
    # Decide what experiments to run.
    configs = []
    for model in args.model.split(","):
        for seed in args.seed.split(","):
            config = ["--seed={}".format(seed), "--model={}".format(model),
                      "--num-steps={}".format(args.num_steps)]
            if args.cuda:
                config.append("--cuda")
            if args.jit:
                config.append("--jit")
                config.append("--time-compilation")
            configs.append(tuple(config))

    # Run timing experiments serially.
    results = {}
    if os.path.exists(args.filename):
        try:
            with open(args.filename, "rb") as f:
                results = pickle.load(f)
        except Exception:
            pass
    for config in configs:
        with timed() as t:
            out = subprocess.check_output((sys.executable, "-O", abspath(join(EXAMPLES_DIR, "hmm.py"))) + config,
                                          encoding="utf-8")
        results[config] = t.elapsed
        if "--jit" in config:
            matched = re.search(r"time to compile: (\d+\.\d+)", out)
            if matched:
                compilation_time = float(matched.group(1))
                results[config + ("(compilation time)",)] = compilation_time
        with open(args.filename, "wb") as f:
            pickle.dump(results, f)

    # Group by seed.
    grouped = defaultdict(list)
    for config, elapsed in results.items():
        grouped[config[1:]].append(elapsed)

    # Print a table in github markdown format.
    print("| Min (sec) | Mean (sec) | Max (sec) | python -O examples/hmm.py ... |")
    print("| -: | -: | -: | - |")
    for config, times in sorted(grouped.items()):
        print("| {:0.1f} | {:0.1f} | {:0.1f} | {} |".format(
            min(times), median(times), max(times), " ".join(config)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profiler for examples/hmm.py")
    parser.add_argument("-f", "--filename", default="hmm_profile.pkl")
    parser.add_argument("-n", "--num-steps", default=50, type=int)
    parser.add_argument("-s", "--seed", default="0,1,2,3,4")
    parser.add_argument("-m", "--model", default="1,2,3,4,5,6,7")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--jit", action="store_true")
    args = parser.parse_args()
    main(args)
