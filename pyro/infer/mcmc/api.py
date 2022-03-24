# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This module offers a modified interface for MCMC inference with the following objectives:
  - making MCMC independent of Pyro specific trace data structure, to facilitate
    integration with other PyTorch based libraries.
  - bringing the interface closer to that of NumPyro to make it easier to write
    code that works with different backends.
  - minimal memory consumption with multiprocessing and CUDA.
"""
import copy
import json
import logging
import queue
import signal
import threading
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict

import torch
import torch.multiprocessing as mp

import pyro
import pyro.poutine as poutine
from pyro.infer.mcmc.hmc import HMC
from pyro.infer.mcmc.logger import (
    DIAGNOSTIC_MSG,
    ProgressBar,
    TqdmHandler,
    initialize_logger,
)
from pyro.infer.mcmc.nuts import NUTS
from pyro.infer.mcmc.util import (
    diagnostics,
    diagnostics_from_stats,
    print_summary,
    select_samples,
)
from pyro.ops.streaming import CountMeanVarianceStats, StatsOfDict, StreamingStats
from pyro.util import optional

MAX_SEED = 2**32 - 1


def logger_thread(
    log_queue, warmup_steps, num_samples, num_chains, disable_progbar=False
):
    """
    Logging thread that asynchronously consumes logging events from `log_queue`,
    and handles them appropriately.
    """
    progress_bars = ProgressBar(
        warmup_steps, num_samples, disable=disable_progbar, num_bars=num_chains
    )
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.addHandler(TqdmHandler())
    num_samples = [0] * num_chains
    try:
        while True:
            try:
                record = log_queue.get(timeout=1)
            except queue.Empty:
                continue
            if record is None:
                break
            metadata, msg = record.getMessage().split("]", 1)
            _, msg_type, logger_id = metadata[1:].split()
            if msg_type == DIAGNOSTIC_MSG:
                pbar_pos = int(logger_id.split(":")[-1])
                num_samples[pbar_pos] += 1
                if num_samples[pbar_pos] == warmup_steps:
                    progress_bars.set_description(
                        "Sample [{}]".format(pbar_pos + 1), pos=pbar_pos
                    )
                diagnostics = json.loads(msg, object_pairs_hook=OrderedDict)
                progress_bars.set_postfix(diagnostics, pos=pbar_pos, refresh=False)
                progress_bars.update(pos=pbar_pos)
            else:
                logger.handle(record)
    finally:
        progress_bars.close()


class _Worker:
    def __init__(
        self,
        chain_id,
        result_queue,
        log_queue,
        event,
        kernel,
        num_samples,
        warmup_steps,
        initial_params=None,
        hook=None,
    ):
        self.chain_id = chain_id
        self.kernel = kernel
        if initial_params is not None:
            self.kernel.initial_params = initial_params
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.rng_seed = (torch.initial_seed() + chain_id) % MAX_SEED
        self.log_queue = log_queue
        self.result_queue = result_queue
        self.default_tensor_type = torch.Tensor().type()
        self.hook = hook
        self.event = event

    def run(self, *args, **kwargs):
        pyro.set_rng_seed(self.rng_seed)
        torch.set_default_tensor_type(self.default_tensor_type)
        kwargs = kwargs
        logger = logging.getLogger("pyro.infer.mcmc")
        logger_id = "CHAIN:{}".format(self.chain_id)
        log_queue = self.log_queue
        logger = initialize_logger(logger, logger_id, None, log_queue)
        logging_hook = _add_logging_hook(logger, None, self.hook)

        try:
            for sample in _gen_samples(
                self.kernel,
                self.warmup_steps,
                self.num_samples,
                logging_hook,
                None,
                *args,
                **kwargs
            ):
                self.result_queue.put_nowait((self.chain_id, sample))
                self.event.wait()
                self.event.clear()
            self.result_queue.put_nowait((self.chain_id, None))
        except Exception as e:
            logger.exception(e)
            self.result_queue.put_nowait((self.chain_id, e))


def _gen_samples(kernel, warmup_steps, num_samples, hook, chain_id, *args, **kwargs):
    kernel.setup(warmup_steps, *args, **kwargs)
    params = kernel.initial_params
    save_params = getattr(kernel, "save_params", sorted(params))
    # yield structure (key, value.shape) of params
    yield {name: params[name].shape for name in save_params}
    for i in range(warmup_steps):
        params = kernel.sample(params)
        hook(
            kernel,
            params,
            "Warmup [{}]".format(chain_id) if chain_id is not None else "Warmup",
            i,
        )
    for i in range(num_samples):
        params = kernel.sample(params)
        hook(
            kernel,
            params,
            "Sample [{}]".format(chain_id) if chain_id is not None else "Sample",
            i,
        )
        flat = [params[name].reshape(-1) for name in save_params]
        yield (torch.cat if flat else torch.tensor)(flat)
    yield kernel.diagnostics()
    kernel.cleanup()


def _add_logging_hook(logger, progress_bar=None, hook=None):
    def _add_logging(kernel, params, stage, i):
        diagnostics = json.dumps(kernel.logging())
        logger.info(diagnostics, extra={"msg_type": DIAGNOSTIC_MSG})
        if progress_bar:
            progress_bar.set_description(stage, refresh=False)
        if hook:
            hook(kernel, params, stage, i)

    return _add_logging


class _UnarySampler:
    """
    Single process runner class optimized for the case chains are drawn sequentially.
    """

    def __init__(
        self,
        kernel,
        num_samples,
        warmup_steps,
        num_chains,
        disable_progbar,
        initial_params=None,
        hook=None,
    ):
        self.kernel = kernel
        self.initial_params = initial_params
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.logger = None
        self.disable_progbar = disable_progbar
        self.hook = hook
        super().__init__()

    def terminate(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        logger = logging.getLogger("pyro.infer.mcmc")
        for i in range(self.num_chains):
            if self.initial_params is not None:
                initial_params = {k: v[i] for k, v in self.initial_params.items()}
                self.kernel.initial_params = initial_params

            progress_bar = ProgressBar(
                self.warmup_steps, self.num_samples, disable=self.disable_progbar
            )
            logger = initialize_logger(logger, "", progress_bar)
            hook_w_logging = _add_logging_hook(logger, progress_bar, self.hook)
            for sample in _gen_samples(
                self.kernel,
                self.warmup_steps,
                self.num_samples,
                hook_w_logging,
                i if self.num_chains > 1 else None,
                *args,
                **kwargs
            ):
                yield sample, i  # sample, chain_id
            self.kernel.cleanup()
            progress_bar.close()


class _MultiSampler:
    """
    Parallel runner class for running MCMC chains in parallel. This uses the
    `torch.multiprocessing` module (itself a light wrapper over the python
    `multiprocessing` module) to spin up parallel workers.
    """

    def __init__(
        self,
        kernel,
        num_samples,
        warmup_steps,
        num_chains,
        mp_context,
        disable_progbar,
        initial_params=None,
        hook=None,
    ):
        self.kernel = kernel
        self.warmup_steps = warmup_steps
        self.num_chains = num_chains
        self.hook = hook
        self.workers = []
        self.ctx = mp
        if mp_context:
            self.ctx = mp.get_context(mp_context)
        self.result_queue = self.ctx.Queue()
        self.log_queue = self.ctx.Queue()
        self.logger = initialize_logger(
            logging.getLogger("pyro.infer.mcmc"), "MAIN", log_queue=self.log_queue
        )
        self.num_samples = num_samples
        self.initial_params = initial_params
        self.log_thread = threading.Thread(
            target=logger_thread,
            args=(
                self.log_queue,
                self.warmup_steps,
                self.num_samples,
                self.num_chains,
                disable_progbar,
            ),
        )
        self.log_thread.daemon = True
        self.log_thread.start()
        self.events = [self.ctx.Event() for _ in range(num_chains)]

    def init_workers(self, *args, **kwargs):
        self.workers = []
        for i in range(self.num_chains):
            init_params = (
                {k: v[i] for k, v in self.initial_params.items()}
                if self.initial_params is not None
                else None
            )
            worker = _Worker(
                i,
                self.result_queue,
                self.log_queue,
                self.events[i],
                self.kernel,
                self.num_samples,
                self.warmup_steps,
                initial_params=init_params,
                hook=self.hook,
            )
            worker.daemon = True
            self.workers.append(
                self.ctx.Process(
                    name=str(i), target=worker.run, args=args, kwargs=kwargs
                )
            )

    def terminate(self, terminate_workers=False):
        if self.log_thread.is_alive():
            self.log_queue.put_nowait(None)
            self.log_thread.join(timeout=1)
        # Only kill workers if exception is raised. worker processes are daemon
        # processes that will otherwise be terminated with the main process.
        # Note that it is important to not
        if terminate_workers:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()

    def run(self, *args, **kwargs):
        # Ignore sigint in worker processes; they will be shut down
        # when the main process terminates.
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.init_workers(*args, **kwargs)
        # restore original handler
        signal.signal(signal.SIGINT, sigint_handler)
        active_workers = self.num_chains
        exc_raised = True
        try:
            for w in self.workers:
                w.start()
            while active_workers:
                try:
                    chain_id, val = self.result_queue.get(timeout=5)
                except queue.Empty:
                    continue
                if isinstance(val, Exception):
                    # Exception trace is already logged by worker.
                    raise val
                if val is not None:
                    yield val, chain_id
                    self.events[chain_id].set()
                else:
                    active_workers -= 1
            exc_raised = False
        finally:
            self.terminate(terminate_workers=exc_raised)


class AbstractMCMC(ABC):
    """
    Base class for MCMC methods.
    """

    def __init__(self, kernel, num_chains, transforms):
        self.kernel = kernel
        self.num_chains = num_chains
        self.transforms = transforms

    @abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def diagnostics(self):
        raise NotImplementedError

    def _set_transforms(self, *args, **kwargs):
        # Use `kernel.transforms` when available
        if getattr(self.kernel, "transforms", None) is not None:
            self.transforms = self.kernel.transforms
        # Else, get transforms from model (e.g. in multiprocessing).
        elif self.kernel.model:
            warmup_steps = 0
            self.kernel.setup(warmup_steps, *args, **kwargs)
            self.transforms = self.kernel.transforms
        # Assign default value
        else:
            self.transforms = {}

    def _validate_kernel(self, initial_params):
        if (
            isinstance(self.kernel, (HMC, NUTS))
            and self.kernel.potential_fn is not None
        ):
            if initial_params is None:
                raise ValueError(
                    "Must provide valid initial parameters to begin sampling"
                    " when using `potential_fn` in HMC/NUTS kernel."
                )

    def _validate_initial_params(self, initial_params):
        for v in initial_params.values():
            if v.shape[0] != self.num_chains:
                raise ValueError(
                    "The leading dimension of tensors in `initial_params` "
                    "must match the number of chains."
                )


class MCMC(AbstractMCMC):
    """
    Wrapper class for Markov Chain Monte Carlo algorithms. Specific MCMC algorithms
    are TraceKernel instances and need to be supplied as a ``kernel`` argument
    to the constructor.

    .. note:: The case of `num_chains > 1` uses python multiprocessing to
        run parallel chains in multiple processes. This goes with the usual
        caveats around multiprocessing in python, e.g. the model used to
        initialize the ``kernel`` must be serializable via `pickle`, and the
        performance / constraints will be platform dependent (e.g. only
        the "spawn" context is available in Windows). This has also not
        been extensively tested on the Windows platform.

    :param kernel: An instance of the ``TraceKernel`` class, which when
        given an execution trace returns another sample trace from the target
        (posterior) distribution.
    :param int num_samples: The number of samples that need to be generated,
        excluding the samples discarded during the warmup phase.
    :param int warmup_steps: Number of warmup iterations. The samples generated
        during the warmup phase are discarded. If not provided, default is
        is the same as `num_samples`.
    :param int num_chains: Number of MCMC chains to run in parallel. Depending on
        whether `num_chains` is 1 or more than 1, this class internally dispatches
        to either `_UnarySampler` or `_MultiSampler`.
    :param dict initial_params: dict containing initial tensors in unconstrained
        space to initiate the markov chain. The leading dimension's size must match
        that of `num_chains`. If not specified, parameter values will be sampled from
        the prior.
    :param hook_fn: Python callable that takes in `(kernel, samples, stage, i)`
        as arguments. stage is either `sample` or `warmup` and i refers to the
        i'th sample for the given stage. This can be used to implement additional
        logging, or more generally, run arbitrary code per generated sample.
    :param str mp_context: Multiprocessing context to use when `num_chains > 1`.
        Only applicable for Python 3.5 and above. Use `mp_context="spawn"` for
        CUDA.
    :param bool disable_progbar: Disable progress bar and diagnostics update.
    :param bool disable_validation: Disables distribution validation check.
        Defaults to ``True``, disabling validation, since divergent transitions
        will lead to exceptions. Switch to ``False`` to enable validation, or
        to ``None`` to preserve existing global values.
    :param dict transforms: dictionary that specifies a transform for a sample site
        with constrained support to unconstrained space.
    :param List[str] save_params: Optional list of a subset of parameter names to
        save during sampling and diagnostics. This is useful in models with
        large nuisance variables. Defaults to None, saving all params.
    """

    def __init__(
        self,
        kernel,
        num_samples,
        warmup_steps=None,
        initial_params=None,
        num_chains=1,
        hook_fn=None,
        mp_context=None,
        disable_progbar=False,
        disable_validation=True,
        transforms=None,
        save_params=None,
    ):
        super().__init__(kernel, num_chains, transforms)
        self.warmup_steps = (
            num_samples if warmup_steps is None else warmup_steps
        )  # Stan
        self.num_samples = num_samples
        self.disable_validation = disable_validation
        self._samples = None
        self._args = None
        self._kwargs = None
        if save_params is not None:
            kernel.save_params = save_params
        self._validate_kernel(initial_params)
        parallel = False
        if num_chains > 1:
            # check that initial_params is different for each chain
            if initial_params:
                self._validate_initial_params(initial_params)
                # FIXME: probably we want to use "spawn" method by default to avoid the error
                # CUDA initialization error https://github.com/pytorch/pytorch/issues/2517
                # even that we run MCMC in CPU.
                if mp_context is None:
                    # change multiprocessing context to 'spawn' for CUDA tensors.
                    if list(initial_params.values())[0].is_cuda:
                        mp_context = "spawn"

            # verify num_chains is compatible with available CPU.
            available_cpu = max(
                mp.cpu_count() - 1, 1
            )  # reserving 1 for the main process.
            if num_chains <= available_cpu:
                parallel = True
            else:
                warnings.warn(
                    "num_chains={} is more than available_cpu={}. "
                    "Chains will be drawn sequentially.".format(
                        num_chains, available_cpu
                    )
                )
        else:
            if initial_params:
                initial_params = {k: v.unsqueeze(0) for k, v in initial_params.items()}
        self._diagnostics = [None] * num_chains
        if parallel:
            self.sampler = _MultiSampler(
                kernel,
                num_samples,
                self.warmup_steps,
                num_chains,
                mp_context,
                disable_progbar,
                initial_params=initial_params,
                hook=hook_fn,
            )
        else:
            self.sampler = _UnarySampler(
                kernel,
                num_samples,
                self.warmup_steps,
                num_chains,
                disable_progbar,
                initial_params=initial_params,
                hook=hook_fn,
            )

    @poutine.block
    def run(self, *args, **kwargs):
        """
        Run MCMC to generate samples and populate `self._samples`.

        Example usage:

        .. code-block:: python

            def model(data):
                ...

            nuts_kernel = NUTS(model)
            mcmc = MCMC(nuts_kernel, num_samples=500)
            mcmc.run(data)
            samples = mcmc.get_samples()

        :param args: optional arguments taken by
            :meth:`MCMCKernel.setup <pyro.infer.mcmc.mcmc_kernel.MCMCKernel.setup>`.
        :param kwargs: optional keywords arguments taken by
            :meth:`MCMCKernel.setup <pyro.infer.mcmc.mcmc_kernel.MCMCKernel.setup>`.
        """
        self._args, self._kwargs = args, kwargs
        num_samples = [0] * self.num_chains
        z_flat_acc = [[] for _ in range(self.num_chains)]
        with optional(
            pyro.validation_enabled(not self.disable_validation),
            self.disable_validation is not None,
        ):
            # XXX we clone CUDA tensor args to resolve the issue "Invalid device pointer"
            # at https://github.com/pytorch/pytorch/issues/10375
            # This also resolves "RuntimeError: Cowardly refusing to serialize non-leaf tensor which
            # requires_grad", which happens with `jit_compile` under PyTorch 1.7
            args = [arg.detach() if torch.is_tensor(arg) else arg for arg in args]
            for x, chain_id in self.sampler.run(*args, **kwargs):
                if num_samples[chain_id] == 0:
                    num_samples[chain_id] += 1
                    z_structure = x
                elif num_samples[chain_id] == self.num_samples + 1:
                    self._diagnostics[chain_id] = x
                else:
                    num_samples[chain_id] += 1
                    if self.num_chains > 1:
                        x_cloned = x.clone()
                        del x
                    else:
                        x_cloned = x
                    z_flat_acc[chain_id].append(x_cloned)

        z_flat_acc = torch.stack([torch.stack(l) for l in z_flat_acc])

        # unpack latent
        pos = 0
        z_acc = z_structure.copy()
        for k in sorted(z_structure):
            shape = z_structure[k]
            next_pos = pos + shape.numel()
            z_acc[k] = z_flat_acc[:, :, pos:next_pos].reshape(
                (self.num_chains, self.num_samples) + shape
            )
            pos = next_pos
        assert pos == z_flat_acc.shape[-1]

        # If transforms is not explicitly provided, infer automatically using
        # model args, kwargs.
        if self.transforms is None:
            self._set_transforms(*args, **kwargs)

        # transform samples back to constrained space
        for name, z in z_acc.items():
            if name in self.transforms:
                z_acc[name] = self.transforms[name].inv(z)
        self._samples = z_acc

        # terminate the sampler (shut down worker processes)
        self.sampler.terminate(True)

    def get_samples(self, num_samples=None, group_by_chain=False):
        """
        Get samples from the MCMC run, potentially resampling with replacement.

        For parameter details see: :meth:`select_samples <pyro.infer.mcmc.util.select_samples>`.
        """
        samples = self._samples
        return select_samples(samples, num_samples, group_by_chain)

    def diagnostics(self):
        """
        Gets some diagnostics statistics such as effective sample size, split
        Gelman-Rubin, or divergent transitions from the sampler.
        """
        diag = diagnostics(self._samples)
        for diag_name in self._diagnostics[0]:
            diag[diag_name] = {
                "chain {}".format(i): self._diagnostics[i][diag_name]
                for i in range(self.num_chains)
            }
        return diag

    def summary(self, prob=0.9):
        """
        Prints a summary table displaying diagnostics of samples obtained from
        posterior. The diagnostics displayed are mean, standard deviation, median,
        the 90% Credibility Interval, :func:`~pyro.ops.stats.effective_sample_size`,
        :func:`~pyro.ops.stats.split_gelman_rubin`.

        :param float prob: the probability mass of samples within the credibility interval.
        """
        print_summary(self._samples, prob=prob)
        if "divergences" in self._diagnostics[0]:
            print(
                "Number of divergences: {}".format(
                    sum(
                        [
                            len(self._diagnostics[i]["divergences"])
                            for i in range(self.num_chains)
                        ]
                    )
                )
            )


class StreamingMCMC(AbstractMCMC):
    """
    MCMC that computes required statistics in a streaming fashion. For this class no samples are retained
    but only aggregated statistics. This is useful for running memory expensive models where we care only
    about specific statistics (especially useful in a memory constrained environments like GPU).

    For available streaming ops please see :mod:`~pyro.ops.streaming`.
    """

    def __init__(
        self,
        kernel,
        num_samples,
        warmup_steps=None,
        initial_params=None,
        statistics=None,
        num_chains=1,
        hook_fn=None,
        disable_progbar=False,
        disable_validation=True,
        transforms=None,
        save_params=None,
    ):
        super().__init__(kernel, num_chains, transforms)
        self.warmup_steps = (
            num_samples if warmup_steps is None else warmup_steps
        )  # Stan
        self.num_samples = num_samples
        self.disable_validation = disable_validation
        self._samples = None
        self._args = None
        self._kwargs = None
        if statistics is None:
            statistics = StatsOfDict(default=CountMeanVarianceStats)
        self._statistics = statistics
        self._default_statistics = copy.deepcopy(statistics)
        if save_params is not None:
            kernel.save_params = save_params
        self._validate_kernel(initial_params)
        if num_chains > 1:
            if initial_params:
                self._validate_initial_params(initial_params)
        else:
            if initial_params:
                initial_params = {k: v.unsqueeze(0) for k, v in initial_params.items()}
        self._diagnostics = [None] * num_chains
        self.sampler = _UnarySampler(
            kernel,
            num_samples,
            self.warmup_steps,
            num_chains,
            disable_progbar,
            initial_params=initial_params,
            hook=hook_fn,
        )

    @poutine.block
    def run(self, *args, **kwargs):
        """
        Run StreamingMCMC to compute required `self._statistics`.
        """
        self._args, self._kwargs = args, kwargs
        num_samples = [0] * self.num_chains

        with optional(
            pyro.validation_enabled(not self.disable_validation),
            self.disable_validation is not None,
        ):
            args = [arg.detach() if torch.is_tensor(arg) else arg for arg in args]
            for x, chain_id in self.sampler.run(*args, **kwargs):
                if num_samples[chain_id] == 0:
                    # If transforms is not explicitly provided, infer automatically using
                    # model args, kwargs.
                    if self.transforms is None:
                        self._set_transforms(*args, **kwargs)
                    num_samples[chain_id] += 1
                    z_structure = x
                elif num_samples[chain_id] == self.num_samples + 1:
                    self._diagnostics[chain_id] = x
                else:
                    num_samples[chain_id] += 1
                    if self.num_chains > 1:
                        x_cloned = x.clone()
                        del x
                    else:
                        x_cloned = x

                    # unpack latent
                    pos = 0
                    z_acc = z_structure.copy()
                    for k in sorted(z_structure):
                        shape = z_structure[k]
                        next_pos = pos + shape.numel()
                        z_acc[k] = x_cloned[pos:next_pos].reshape(shape)
                        pos = next_pos

                    for name, z in z_acc.items():
                        if name in self.transforms:
                            z_acc[name] = self.transforms[name].inv(z)

                    self._statistics.update(
                        {
                            (chain_id, name): transformed_sample
                            for name, transformed_sample in z_acc.items()
                        }
                    )

        # terminate the sampler (shut down worker processes)
        self.sampler.terminate(True)

    def get_statistics(self, group_by_chain=True):
        """
        Returns a dict of statistics defined by those passed to the class constructor.

        :param bool group_by_chain: Whether statistics should be chain-wise or merged together.
        """
        if group_by_chain:
            return self._statistics.get()
        else:
            # merge all chains with respect to names
            merged_dict: Dict[str, StreamingStats] = {}
            for (_, name), stat in self._statistics.stats.items():
                if name in merged_dict:
                    merged_dict[name] = merged_dict[name].merge(stat)
                else:
                    merged_dict[name] = stat

        return {k: v.get() for k, v in merged_dict.items()}

    def diagnostics(self):
        """
        Gets diagnostics. Currently a split Gelman-Rubin is only supported and requires
        'mean' and 'variance' streaming statistics to be present.
        """
        statistics = self._statistics.get()
        diag = diagnostics_from_stats(statistics, self.num_samples, self.num_chains)
        for diag_name in self._diagnostics[0]:
            diag[diag_name] = {
                "chain {}".format(i): self._diagnostics[i][diag_name]
                for i in range(self.num_chains)
            }
        return diag
