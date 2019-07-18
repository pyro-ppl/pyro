import argparse
import os
import pickle
import subprocess
import sys
import timeit
from collections import defaultdict

from numpy import median


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
        start_time = timeit.default_timer()
        subprocess.check_call((sys.executable, "-O", "examples/hmm.py") + config)
        elapsed = timeit.default_timer() - start_time
        results[config] = elapsed
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
