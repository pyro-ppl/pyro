# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import torch

import pyro
from pyro.contrib.epidemiology.models import RegionalSIRModel

logging.basicConfig(format='%(message)s', level=logging.INFO)


def Model(args, data):
    assert 0 <= args.coupling <= 1, args.coupling
    population = torch.full((args.num_regions,), float(args.population))
    coupling = torch.eye(args.num_regions).clamp(min=args.coupling)
    return RegionalSIRModel(population, coupling, args.recovery_time, data)


def generate_data(args):
    extended_data = [None] * (args.duration + args.forecast)
    model = Model(args, extended_data)
    logging.info("Simulating from a {}".format(type(model).__name__))
    for attempt in range(100):
        samples = model.generate({"R0": args.basic_reproduction_number,
                                  "rho_c1": 10 * args.response_rate,
                                  "rho_c0": 10 * (1 - args.response_rate)})
        obs = samples["obs"][:args.duration]
        S2I = samples["S2I"]

        obs_sum = int(obs.sum())
        S2I_sum = int(S2I[:args.duration].sum())
        if obs_sum >= args.min_observations:
            logging.info("Observed {:d}/{:d} infections:\n{}".format(
                obs_sum, S2I_sum, " ".join(str(int(x)) for x in obs[:, 0])))
            return {"S2I": S2I, "obs": obs}

    raise ValueError("Failed to generate {} observations. Try increasing "
                     "--population or decreasing --min-observations"
                     .format(args.min_observations))


def infer_mcmc(args, model):
    energies = []

    def hook_fn(kernel, *unused):
        e = float(kernel._potential_energy_last)
        energies.append(e)
        if args.verbose:
            logging.info("potential = {:0.6g}".format(e))

    mcmc = model.fit_mcmc(heuristic_num_particles=args.smc_particles,
                          heuristic_ess_threshold=args.ess_threshold,
                          warmup_steps=args.warmup_steps,
                          num_samples=args.num_samples,
                          max_tree_depth=args.max_tree_depth,
                          num_quant_bins=args.num_bins,
                          haar=args.haar,
                          haar_full_mass=args.haar_full_mass,
                          jit_compile=args.jit,
                          hook_fn=hook_fn)

    mcmc.summary()
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.plot(energies)
        plt.xlabel("MCMC step")
        plt.ylabel("potential energy")
        plt.title("MCMC energy trace")
        plt.tight_layout()


def infer_svi(args, model):
    losses = model.fit_svi(heuristic_num_particles=args.smc_particles,
                           heuristic_ess_threshold=args.ess_threshold,
                           num_samples=args.num_samples,
                           num_steps=args.svi_steps,
                           num_particles=args.svi_particles,
                           haar=args.haar,
                           jit=args.jit)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.plot(losses)
        plt.xlabel("SVI step")
        plt.ylabel("loss")
        plt.title("SVI Convergence")
        plt.tight_layout()


def predict(args, model, truth):
    samples = model.predict(forecast=args.forecast)
    S2I = samples["S2I"]
    median = S2I.median(dim=0).values
    lines = ["Median prediction of new infections (starting on day 0):"]
    for r in range(args.num_regions):
        lines.append("Region {}: {}".format(r, " ".join(map(str, map(int, median[:, r])))))
    logging.info("\n".join(lines))

    # Optionally plot the latent and forecasted series of new infections.
    if args.plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(args.num_regions, sharex=True,
                                 figsize=(6, 1 + args.num_regions))
        time = torch.arange(args.duration + args.forecast)
        p05 = S2I.kthvalue(int(round(0.5 + 0.05 * args.num_samples)), dim=0).values
        p95 = S2I.kthvalue(int(round(0.5 + 0.95 * args.num_samples)), dim=0).values
        for r, ax in enumerate(axes):
            ax.fill_between(time, p05[:, r], p95[:, r], color="red", alpha=0.3, label="90% CI")
            ax.plot(time, median[:, r], "r-", label="median")
            ax.plot(time[:args.duration], model.data[:, r], "k.", label="observed")
            ax.plot(time, truth[:, r], "k--", label="truth")
            ax.axvline(args.duration - 0.5, color="gray", lw=1)
            ax.set_xlim(0, len(time) - 1)
            ax.set_ylim(0, None)
        axes[0].set_title("New infections among {} regions each of size {}"
                          .format(args.num_regions, args.population))
        axes[args.num_regions // 2].set_ylabel("inf./day")
        axes[-1].set_xlabel("day after first infection")
        axes[-1].legend(loc="upper left")
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)


def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.rng_seed)

    # Generate data.
    dataset = generate_data(args)
    obs = dataset["obs"]

    # Run inference.
    model = Model(args, obs)
    infer = {"mcmc": infer_mcmc, "svi": infer_svi}[args.infer]
    infer(args, model)

    # Predict latent time series.
    predict(args, model, truth=dataset["S2I"])


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(
        description="Regional compartmental epidemiology modeling using HMC")
    parser.add_argument("-p", "--population", default=1000, type=int)
    parser.add_argument("-r", "--num-regions", default=2, type=int)
    parser.add_argument("-c", "--coupling", default=0.1, type=float)
    parser.add_argument("-m", "--min-observations", default=3, type=int)
    parser.add_argument("-d", "--duration", default=20, type=int)
    parser.add_argument("-f", "--forecast", default=10, type=int)
    parser.add_argument("-R0", "--basic-reproduction-number", default=1.5, type=float)
    parser.add_argument("-tau", "--recovery-time", default=7.0, type=float)
    parser.add_argument("-rho", "--response-rate", default=0.5, type=float)
    parser.add_argument("--infer", default="mcmc")
    parser.add_argument("--mcmc", action="store_const", const="mcmc", dest="infer")
    parser.add_argument("--svi", action="store_const", const="svi", dest="infer")
    parser.add_argument("--haar", action="store_true")
    parser.add_argument("-hfm", "--haar-full-mass", default=0, type=int)
    parser.add_argument("-n", "--num-samples", default=200, type=int)
    parser.add_argument("-np", "--smc-particles", default=1024, type=int)
    parser.add_argument("-ss", "--svi-steps", default=5000, type=int)
    parser.add_argument("-sp", "--svi-particles", default=32, type=int)
    parser.add_argument("-ess", "--ess-threshold", default=0.5, type=float)
    parser.add_argument("-w", "--warmup-steps", type=int)
    parser.add_argument("-t", "--max-tree-depth", default=5, type=int)
    parser.add_argument("-nb", "--num-bins", default=1, type=int)
    parser.add_argument("--double", action="store_true", default=True)
    parser.add_argument("--single", action="store_false", dest="double")
    parser.add_argument("--rng-seed", default=0, type=int)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--jit", action="store_true", default=True)
    parser.add_argument("--nojit", action="store_false", dest="jit")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    if args.warmup_steps is None:
        args.warmup_steps = args.num_samples
    if args.double:
        if args.cuda:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_dtype(torch.float64)
    elif args.cuda:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    main(args)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.show()
