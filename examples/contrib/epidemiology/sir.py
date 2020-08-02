# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

# This script aims to replicate the behavior of examples/sir_hmc.py but using
# the high-level components of pyro.contrib.epidemiology. Command line
# arguments and results should be similar.

import argparse
import logging
import math

import torch
from torch.distributions import biject_to, constraints

import pyro
from pyro.contrib.epidemiology.models import (HeterogeneousSIRModel, OverdispersedSEIRModel, OverdispersedSIRModel,
                                              SimpleSEIRModel, SimpleSIRModel, SuperspreadingSEIRModel,
                                              SuperspreadingSIRModel)

logging.basicConfig(format='%(message)s', level=logging.INFO)


def Model(args, data):
    """Dispatch between different model classes."""
    if args.heterogeneous:
        assert args.incubation_time == 0
        assert args.overdispersion == 0
        return HeterogeneousSIRModel(args.population, args.recovery_time, data)
    elif args.incubation_time > 0:
        assert args.incubation_time > 1
        if args.concentration < math.inf:
            return SuperspreadingSEIRModel(args.population, args.incubation_time,
                                           args.recovery_time, data)
        elif args.overdispersion > 0:
            return OverdispersedSEIRModel(args.population, args.incubation_time,
                                          args.recovery_time, data)
        else:
            return SimpleSEIRModel(args.population, args.incubation_time,
                                   args.recovery_time, data)
    else:
        if args.concentration < math.inf:
            return SuperspreadingSIRModel(args.population, args.recovery_time, data)
        elif args.overdispersion > 0:
            return OverdispersedSIRModel(args.population, args.recovery_time, data)
        else:
            return SimpleSIRModel(args.population, args.recovery_time, data)


def generate_data(args):
    extended_data = [None] * (args.duration + args.forecast)
    model = Model(args, extended_data)
    logging.info("Simulating from a {}".format(type(model).__name__))
    for attempt in range(100):
        samples = model.generate({"R0": args.basic_reproduction_number,
                                  "rho": args.response_rate,
                                  "k": args.concentration,
                                  "od": args.overdispersion})
        obs = samples["obs"][:args.duration]
        new_I = samples.get("S2I", samples.get("E2I"))

        obs_sum = int(obs.sum())
        new_I_sum = int(new_I[:args.duration].sum())
        assert 0 <= args.min_obs_portion < args.max_obs_portion <= 1
        min_obs = int(math.ceil(args.min_obs_portion * args.population))
        max_obs = int(math.floor(args.max_obs_portion * args.population))
        if min_obs <= obs_sum <= max_obs:
            logging.info("Observed {:d}/{:d} infections:\n{}".format(
                obs_sum, new_I_sum, " ".join(str(int(x)) for x in obs)))
            return {"new_I": new_I, "obs": obs}

    if obs_sum < min_obs:
        raise ValueError("Failed to generate >={} observations. "
                         "Try decreasing --min-obs-portion (currently {})."
                         .format(min_obs, args.min_obs_portion))
    else:
        raise ValueError("Failed to generate <={} observations. "
                         "Try increasing --max-obs-portion (currently {})."
                         .format(max_obs, args.max_obs_portion))


def infer_mcmc(args, model):
    parallel = args.num_chains > 1
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
                          num_chains=args.num_chains,
                          mp_context="spawn" if parallel else None,
                          max_tree_depth=args.max_tree_depth,
                          arrowhead_mass=args.arrowhead_mass,
                          num_quant_bins=args.num_bins,
                          haar=args.haar,
                          haar_full_mass=args.haar_full_mass,
                          jit_compile=args.jit,
                          hook_fn=None if parallel else hook_fn)

    mcmc.summary()
    if args.plot and energies:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 3))
        plt.plot(energies)
        plt.xlabel("MCMC step")
        plt.ylabel("potential energy")
        plt.title("MCMC energy trace")
        plt.tight_layout()

    return model.samples


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

    return model.samples


def evaluate(args, model, samples):
    # Print estimated values.
    names = {"basic_reproduction_number": "R0"}
    if not args.heterogeneous:
        names["response_rate"] = "rho"
    if args.concentration < math.inf:
        names["concentration"] = "k"
    if "od" in samples:
        names["overdispersion"] = "od"
    for name, key in names.items():
        mean = samples[key].mean().item()
        std = samples[key].std().item()
        logging.info("{}: truth = {:0.3g}, estimate = {:0.3g} \u00B1 {:0.3g}"
                     .format(key, getattr(args, name), mean, std))

    # Optionally plot histograms and pairwise correlations.
    if args.plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Plot individual histograms.
        fig, axes = plt.subplots(len(names), 1, figsize=(5, 2.5 * len(names)))
        if len(names) == 1:
            axes = [axes]
        axes[0].set_title("Posterior parameter estimates")
        for ax, (name, key) in zip(axes, names.items()):
            truth = getattr(args, name)
            sns.distplot(samples[key], ax=ax, label="posterior")
            ax.axvline(truth, color="k", label="truth")
            ax.set_xlabel(key + " = " + name.replace("_", " "))
            ax.set_yticks(())
            ax.legend(loc="best")
        plt.tight_layout()

        # Plot pairwise joint distributions for selected variables.
        covariates = [(name, samples[name]) for name in names.values()]
        for i, aux in enumerate(samples["auxiliary"].squeeze(1).unbind(-2)):
            covariates.append(("aux[{},0]".format(i), aux[:, 0]))
            covariates.append(("aux[{},-1]".format(i), aux[:, -1]))
        N = len(covariates)
        fig, axes = plt.subplots(N, N, figsize=(8, 8), sharex="col", sharey="row")
        for i in range(N):
            axes[i][0].set_ylabel(covariates[i][0])
            axes[0][i].set_xlabel(covariates[i][0])
            axes[0][i].xaxis.set_label_position("top")
            for j in range(N):
                ax = axes[i][j]
                ax.set_xticks(())
                ax.set_yticks(())
                ax.scatter(covariates[j][1], -covariates[i][1],
                           lw=0, color="darkblue", alpha=0.3)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        # Plot Pearson correlation for every pair of unconstrained variables.
        def unconstrain(constraint, value):
            value = biject_to(constraint).inv(value)
            return value.reshape(args.num_samples, -1)

        covariates = [("R1", unconstrain(constraints.positive, samples["R0"]))]
        if not args.heterogeneous:
            covariates.append(
                ("rho", unconstrain(constraints.unit_interval, samples["rho"])))
        if "k" in samples:
            covariates.append(
                ("k", unconstrain(constraints.positive, samples["k"])))
        constraint = constraints.interval(-0.5, model.population + 0.5)
        for name, aux in zip(model.compartments, samples["auxiliary"].unbind(-2)):
            covariates.append((name, unconstrain(constraint, aux)))
        x = torch.cat([v for _, v in covariates], dim=-1)
        x -= x.mean(0)
        x /= x.std(0)
        x = x.t().matmul(x)
        x /= args.num_samples
        x.clamp_(min=-1, max=1)
        plt.figure(figsize=(8, 8))
        plt.imshow(x, cmap="bwr")
        ticks = torch.tensor([0] + [v.size(-1) for _, v in covariates]).cumsum(0)
        ticks = (ticks[1:] + ticks[:-1]) / 2
        plt.yticks(ticks, [name for name, _ in covariates])
        plt.xticks(())
        plt.tick_params(length=0)
        plt.title("Pearson correlation (unconstrained coordinates)")
        plt.tight_layout()


def predict(args, model, truth):
    samples = model.predict(forecast=args.forecast)

    obs = model.data

    new_I = samples.get("S2I", samples.get("E2I"))
    median = new_I.median(dim=0).values
    logging.info("Median prediction of new infections (starting on day 0):\n{}"
                 .format(" ".join(map(str, map(int, median)))))

    # Optionally plot the latent and forecasted series of new infections.
    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        time = torch.arange(args.duration + args.forecast)
        p05 = new_I.kthvalue(int(round(0.5 + 0.05 * args.num_samples)), dim=0).values
        p95 = new_I.kthvalue(int(round(0.5 + 0.95 * args.num_samples)), dim=0).values
        plt.fill_between(time, p05, p95, color="red", alpha=0.3, label="90% CI")
        plt.plot(time, median, "r-", label="median")
        plt.plot(time[:args.duration], obs, "k.", label="observed")
        if truth is not None:
            plt.plot(time, truth, "k--", label="truth")
        plt.axvline(args.duration - 0.5, color="gray", lw=1)
        plt.xlim(0, len(time) - 1)
        plt.ylim(0, None)
        plt.xlabel("day after first infection")
        plt.ylabel("new infections per day")
        plt.title("New infections in population of {}".format(args.population))
        plt.legend(loc="upper left")
        plt.tight_layout()

        # Plot Re time series.
        if args.heterogeneous:
            plt.figure()
            Re = samples["Re"]
            median = Re.median(dim=0).values
            p05 = Re.kthvalue(int(round(0.5 + 0.05 * args.num_samples)), dim=0).values
            p95 = Re.kthvalue(int(round(0.5 + 0.95 * args.num_samples)), dim=0).values
            plt.fill_between(time, p05, p95, color="red", alpha=0.3, label="90% CI")
            plt.plot(time, median, "r-", label="median")
            plt.plot(time[:args.duration], obs, "k.", label="observed")
            plt.axvline(args.duration - 0.5, color="gray", lw=1)
            plt.xlim(0, len(time) - 1)
            plt.ylim(0, None)
            plt.xlabel("day after first infection")
            plt.ylabel("Re")
            plt.title("Effective reproductive number over time")
            plt.legend(loc="upper left")
            plt.tight_layout()


def main(args):
    pyro.enable_validation(__debug__)
    pyro.set_rng_seed(args.rng_seed)

    # Generate data.
    dataset = generate_data(args)
    obs = dataset["obs"]

    # Run inference.
    model = Model(args, obs)
    infer = {"mcmc": infer_mcmc, "svi": infer_svi}[args.infer]
    samples = infer(args, model)

    # Evaluate fit.
    evaluate(args, model, samples)

    # Predict latent time series.
    if args.forecast:
        predict(args, model, truth=dataset["new_I"])


if __name__ == "__main__":
    assert pyro.__version__.startswith('1.4.0')
    parser = argparse.ArgumentParser(
        description="Compartmental epidemiology modeling using HMC")
    parser.add_argument("-p", "--population", default=1000, type=float)
    parser.add_argument("-m", "--min-obs-portion", default=0.01, type=float)
    parser.add_argument("-M", "--max-obs-portion", default=0.99, type=float)
    parser.add_argument("-d", "--duration", default=20, type=int)
    parser.add_argument("-f", "--forecast", default=10, type=int)
    parser.add_argument("-R0", "--basic-reproduction-number", default=1.5, type=float)
    parser.add_argument("-tau", "--recovery-time", default=7.0, type=float)
    parser.add_argument("-e", "--incubation-time", default=0.0, type=float,
                        help="If zero, use SIR model; if > 1 use SEIR model.")
    parser.add_argument("-k", "--concentration", default=math.inf, type=float,
                        help="If finite, use a superspreader model.")
    parser.add_argument("-rho", "--response-rate", default=0.5, type=float)
    parser.add_argument("-o", "--overdispersion", default=0., type=float)
    parser.add_argument("-hg", "--heterogeneous", action="store_true")
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
    parser.add_argument("-c", "--num-chains", default=1, type=int)
    parser.add_argument("-t", "--max-tree-depth", default=5, type=int)
    parser.add_argument("-a", "--arrowhead-mass", action="store_true")
    parser.add_argument("-r", "--rng-seed", default=0, type=int)
    parser.add_argument("-nb", "--num-bins", default=1, type=int)
    parser.add_argument("--double", action="store_true", default=True)
    parser.add_argument("--single", action="store_false", dest="double")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--jit", action="store_true", default=True)
    parser.add_argument("--nojit", action="store_false", dest="jit")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.population = int(args.population)  # to allow e.g. --population=1e6

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
