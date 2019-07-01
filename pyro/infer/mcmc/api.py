"""
This module offers a modified interface for MCMC inference with the following objectives:
  - making MCMC independent of Pyro specific trace data structure, to facilitate
    integration with other PyTorch based libraries.
  - bringing the interface closer to that of NumPyro to make it easier to write
    code that works with different backends.
  - minimal memory consumption with multiprocessing and CUDA.
"""

from __future__ import absolute_import, division, print_function

import json
import logging
import signal
import threading
import warnings
from collections import OrderedDict, defaultdict

import six
import torch
import torch.multiprocessing as mp
from six.moves import queue

import pyro
from pyro.infer.mcmc.logger import initialize_logger, DIAGNOSTIC_MSG, TqdmHandler, ProgressBar

MAX_SEED = 2**32 - 1


def logger_thread(log_queue, warmup_steps, num_samples, num_chains, disable_progbar=False):
    """
    Logging thread that asynchronously consumes logging events from `log_queue`,
    and handles them appropriately.
    """
    progress_bars = ProgressBar(warmup_steps, num_samples, disable=disable_progbar, num_bars=num_chains)
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
                    progress_bars.set_description("Sample [{}]".format(pbar_pos + 1), pos=pbar_pos)
                diagnostics = json.loads(msg, object_pairs_hook=OrderedDict)
                progress_bars.set_postfix(diagnostics, pos=pbar_pos)
                progress_bars.update(pos=pbar_pos)
            else:
                logger.handle(record)
    finally:
        progress_bars.close()


class _Worker(object):
    def __init__(self, chain_id, result_queue, log_queue, event,
                 kernel, num_samples, warmup_steps, initial_params=None, hook=None):
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
        # XXX we clone CUDA tensor args to resolve the issue "Invalid device pointer"
        # at https://github.com/pytorch/pytorch/issues/10375
        args = [arg.clone().detach() if (torch.is_tensor(arg) and arg.is_cuda) else arg for arg in args]
        kwargs = kwargs
        logger = logging.getLogger("pyro.infer.mcmc")
        logger_id = "CHAIN:{}".format(self.chain_id)
        log_queue = self.log_queue
        logger = initialize_logger(logger, logger_id, None, log_queue)
        logging_hook = _add_logging_hook(logger, None, self.hook)

        try:
            for sample in _gen_samples(self.kernel, self.warmup_steps, self.num_samples, logging_hook,
                                       *args, **kwargs):
                self.result_queue.put_nowait((self.chain_id, sample))
                self.event.wait()
                self.event.clear()
            self.result_queue.put_nowait((self.chain_id, None))
        except Exception as e:
            logger.exception(e)
            self.result_queue.put_nowait((self.chain_id, e))


def _gen_samples(kernel, warmup_steps, num_samples, hook, *args, **kwargs):
    kernel.setup(warmup_steps, *args, **kwargs)
    params = kernel.initial_params
    for i in range(warmup_steps):
        params = kernel.sample(params)
        hook(kernel, params, 'warmup', i)
    for i in range(num_samples):
        params = kernel.sample(params)
        hook(kernel, params, 'sample', i)
        yield params
    kernel.cleanup()


def _add_logging_hook(logger, progress_bar=None, hook=None):
    def _add_logging(kernel, params, stage, i):
        diagnostics = json.dumps(kernel.diagnostics())
        logger.info(diagnostics, extra={"msg_type": DIAGNOSTIC_MSG})
        if progress_bar:
            progress_bar.set_description(stage)
        if hook:
            hook(kernel, params, stage, i)

    return _add_logging


class _UnarySampler(object):
    """
    Single process runner class optimized for the case `num_chains=1`.
    """

    def __init__(self, kernel, num_samples, warmup_steps, disable_progbar, initial_params=None, hook=None):
        self.kernel = kernel
        if initial_params is not None:
            self.kernel.initial_params = initial_params
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.logger = None
        self.disable_progbar = disable_progbar
        self.hook = hook
        super(_UnarySampler, self).__init__()

    def run(self, *args, **kwargs):
        logger = logging.getLogger("pyro.infer.mcmc")
        progress_bar = ProgressBar(self.warmup_steps, self.num_samples, disable=self.disable_progbar)
        logger = initialize_logger(logger, "", progress_bar)
        hook_w_logging = _add_logging_hook(logger, progress_bar, self.hook)
        for sample in _gen_samples(self.kernel, self.warmup_steps, self.num_samples, hook_w_logging,
                                   *args, **kwargs):
            yield sample, 0  # sample, chain_id (default=0)
        progress_bar.close()


class _MultiSampler(object):
    """
    Parallel runner class for running MCMC chains in parallel. This uses the
    `torch.multiprocessing` module (itself a light wrapper over the python
    `multiprocessing` module) to spin up parallel workers.
    """
    def __init__(self, kernel, num_samples, warmup_steps, num_chains, mp_context,
                 disable_progbar, initial_params=None, hook=None):
        self.kernel = kernel
        self.warmup_steps = warmup_steps
        self.num_chains = num_chains
        self.hook = hook
        self.workers = []
        self.ctx = mp
        if mp_context:
            if six.PY2:
                raise ValueError("multiprocessing.get_context() is "
                                 "not supported in Python 2.")
            self.ctx = mp.get_context(mp_context)
        self.result_queue = self.ctx.Queue()
        self.log_queue = self.ctx.Queue()
        self.logger = initialize_logger(logging.getLogger("pyro.infer.mcmc"),
                                        "MAIN", log_queue=self.log_queue)
        self.num_samples = num_samples
        self.initial_params = initial_params
        self.log_thread = threading.Thread(target=logger_thread,
                                           args=(self.log_queue, self.warmup_steps, self.num_samples,
                                                 self.num_chains, disable_progbar))
        self.log_thread.daemon = True
        self.log_thread.start()
        self.events = [self.ctx.Event() for _ in range(num_chains)]

    def init_workers(self, *args, **kwargs):
        self.workers = []
        for i in range(self.num_chains):
            init_params = {k: v[i] for k, v in self.initial_params.items()}
            worker = _Worker(i, self.result_queue, self.log_queue, self.events[i], self.kernel,
                             self.num_samples, self.warmup_steps, initial_params=init_params, hook=self.hook)
            worker.daemon = True
            self.workers.append(self.ctx.Process(name=str(i), target=worker.run,
                                                 args=args, kwargs=kwargs))

    def terminate(self):
        if self.log_thread.is_alive():
            self.log_queue.put_nowait(None)
            self.log_thread.join(timeout=1)
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
        try:
            for w in self.workers:
                w.start()
            while active_workers:
                try:
                    chain_id, val = self.result_queue.get(timeout=5)
                    self.events[chain_id].set()
                except queue.Empty:
                    continue
                if isinstance(val, Exception):
                    # Exception trace is already logged by worker.
                    raise val
                if val is not None:
                    yield val, chain_id
                else:
                    active_workers -= 1
        finally:
            self.terminate()


class MCMC(object):
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
        half of `num_samples`.
    :param int num_chains: Number of MCMC chains to run in parallel. Depending on
        whether `num_chains` is 1 or more than 1, this class internally dispatches
        to either `_SingleSampler` or `_ParallelSampler`.
    :param dict initial_params: dict containing initial tensors to initiate
        the markov chain. The leading dimension's size must match that of
        `num_chains`. If not specified, parameter values will be sampled from
        the prior.
    :param hook_fn: Python callable that takes in `(kernel, samples, stage, i)`
        as arguments. stage is either `sample` or `warmup` and i refers to the
        i'th sample for the given stage. This can be
    :param str mp_context: Multiprocessing context to use when `num_chains > 1`.
        Only applicable for Python 3.5 and above. Use `mp_context="spawn"` for
        CUDA.
    :param bool disable_progbar: Disable progress bar and diagnostics update.
    """
    def __init__(self, kernel, num_samples, warmup_steps=None, initial_params=None,
                 num_chains=1, hook_fn=None, mp_context=None, disable_progbar=False, transforms=None):
        self.warmup_steps = num_samples if warmup_steps is None else warmup_steps  # Stan
        self.num_samples = num_samples
        self.kernel = kernel
        self.transforms = transforms or self.kernel.transforms
        if num_chains > 1:
            # old API does not work since we need transforms in the main process.
            if self.transforms is None:
                raise ValueError("`pyro.infer.mcmc.api.MCMC` needs explicit reference"
                                 " to transforms from the real domain to the site's bounded"
                                 " support. Use `initialize_model` utility to convert your model"
                                 " function into a potential function along with the required"
                                 " transforms.")

            # check that initial_params is different for each chain
            if initial_params:
                for v in initial_params.values():
                    if v.shape[0] != num_chains:
                        raise ValueError("The leading dimension of tensors in `initial_params` "
                                         "must match the number of chains.")

            # verify num_chains is compatible with available CPU.
            available_cpu = max(mp.cpu_count() - 1, 1)  # reserving 1 for the main process.
            if num_chains > available_cpu:
                warnings.warn("num_chains={} is more than available_cpu={}. "
                              "Resetting number of chains to available CPU count."
                              .format(num_chains, available_cpu))
                num_chains = available_cpu
        self.num_chains = num_chains

        if num_chains > 1:
            self.sampler = _MultiSampler(kernel, num_samples, self.warmup_steps, num_chains, mp_context,
                                         disable_progbar, initial_params=initial_params, hook=hook_fn)
        else:
            self.sampler = _UnarySampler(kernel, num_samples, self.warmup_steps, disable_progbar,
                                         initial_params=initial_params, hook=hook_fn)

    def run(self, *args, **kwargs):
        z_acc = defaultdict(lambda: [[] for _ in range(self.num_chains)])
        for sample, chain_id in self.sampler.run(*args, **kwargs):
            for k, v in sample.items():
                z_acc[k][chain_id].append(v)
        z_acc = {k: [torch.stack(l) for l in v] for k, v in z_acc.items()}
        z_acc = {k: v[0] if self.num_chains == 1 else torch.stack(v) for k, v in z_acc.items()}

        # transform samples back to constrained space
        if self.transforms:
            for name, transform in self.transforms.items():
                z_acc[name] = transform.inv(z_acc[name])
        return z_acc
