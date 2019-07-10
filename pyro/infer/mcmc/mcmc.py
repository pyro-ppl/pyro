from __future__ import absolute_import, division, print_function

import json
import logging
import signal
import threading
import warnings
from collections import OrderedDict

import six
import torch
import torch.multiprocessing as mp
from six.moves import queue

import pyro
import pyro.ops.stats as stats
from pyro.infer import TracePosterior
from pyro.infer.abstract_infer import Marginals
from pyro.infer.mcmc.logger import initialize_logger, DIAGNOSTIC_MSG, TqdmHandler, ProgressBar
import pyro.poutine as poutine
from pyro.util import optional

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
                progress_bars.set_postfix(diagnostics, pos=pbar_pos, refresh=False)
                progress_bars.update(pos=pbar_pos)
            else:
                logger.handle(record)
    finally:
        progress_bars.close()


class _Worker(object):
    def __init__(self, chain_id, result_queue, log_queue, event,
                 kernel, num_samples, warmup_steps=0,
                 args=None, kwargs=None):
        self.chain_id = chain_id
        self.trace_gen = _SingleSampler(kernel, num_samples=num_samples, warmup_steps=warmup_steps,
                                        disable_progbar=True)
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.rng_seed = (torch.initial_seed() + chain_id) % MAX_SEED
        self.log_queue = log_queue
        self.result_queue = result_queue
        self.default_tensor_type = torch.Tensor().type()
        self.event = event

    def run(self):
        pyro.set_rng_seed(self.rng_seed)
        torch.set_default_tensor_type(self.default_tensor_type)
        # XXX we clone CUDA tensor args to resolve the issue "Invalid device pointer"
        # at https://github.com/pytorch/pytorch/issues/10375
        args = [arg.clone().detach() if (torch.is_tensor(arg) and arg.is_cuda) else arg for arg in self.args]
        kwargs = self.kwargs
        kwargs["logger_id"] = "CHAIN:{}".format(self.chain_id)
        kwargs["log_queue"] = self.log_queue
        try:
            for sample in self.trace_gen._traces(*args, **kwargs):
                self.result_queue.put_nowait((self.chain_id, sample))
                self.event.wait()
                self.event.clear()
            self.result_queue.put_nowait((self.chain_id, None))
        except Exception as e:
            self.trace_gen.logger.exception(e)
            self.result_queue.put_nowait((self.chain_id, e))


class _ParallelSampler(TracePosterior):
    """
    Parallel runner class for running MCMC chains in parallel. This uses the
    `torch.multiprocessing` module (itself a light wrapper over the python
    `multiprocessing` module) to spin up parallel workers.
    """
    def __init__(self, kernel, num_samples, warmup_steps, num_chains, mp_context, disable_progbar):
        super(_ParallelSampler, self).__init__()
        self.kernel = kernel
        self.warmup_steps = warmup_steps
        self.num_chains = num_chains
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
        self.log_thread = threading.Thread(target=logger_thread,
                                           args=(self.log_queue, self.warmup_steps, self.num_samples,
                                                 self.num_chains, disable_progbar))
        self.log_thread.daemon = True
        self.log_thread.start()
        self.events = [self.ctx.Event() for i in range(num_chains)]

    def init_workers(self, *args, **kwargs):
        self.workers = []
        for i in range(self.num_chains):
            worker = _Worker(i, self.result_queue, self.log_queue, self.events[i], self.kernel,
                             self.num_samples, self.warmup_steps, args, kwargs)
            worker.daemon = True
            self.workers.append(self.ctx.Process(name=str(i), target=worker.run))

    def terminate(self):
        if self.log_thread.is_alive():
            self.log_queue.put_nowait(None)
            self.log_thread.join(timeout=1)
        for w in self.workers:
            if w.is_alive():
                w.terminate()

    def _traces(self, *args, **kwargs):
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
                    yield val[0], val[1], chain_id
                else:
                    active_workers -= 1
        finally:
            self.terminate()


class _SingleSampler(TracePosterior):
    """
    Single process runner class optimized for the case `num_chains=1`.
    """
    def __init__(self, kernel, num_samples, warmup_steps, disable_progbar):
        self.kernel = kernel
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.logger = None
        self.disable_progbar = disable_progbar
        super(_SingleSampler, self).__init__()

    def _gen_samples(self, num_samples, init_params):
        params = init_params
        for _ in range(num_samples):
            params = self.kernel.sample(params)
            diagnostics = json.dumps(self.kernel.logging())
            self.logger.info(diagnostics, extra={"msg_type": DIAGNOSTIC_MSG})
            yield params

    # TODO: Refactor so that this class directly has access to the trace generator
    # and transforms needed to do this wrapping. Note that only unconstrained parameters
    # are passed to `MCMCKernel` classes.
    def _trace_wrap(self, z, *args, **kwargs):
        if self.kernel.transforms:
            for name, transform in self.kernel.transforms.items():
                z[name] = transform.inv(z[name])
        if self.kernel.model is None:
            return z
        z_trace = self.kernel._prototype_trace
        for name, value in z.items():
            z_trace.nodes[name]["value"] = value
        trace_poutine = poutine.trace(poutine.replay(self.kernel.model, z_trace))
        trace_poutine(*args, **kwargs)
        return trace_poutine.trace

    def _traces(self, *args, **kwargs):
        logger_id = kwargs.pop("logger_id", "")
        log_queue = kwargs.pop("log_queue", None)
        self.logger = logging.getLogger("pyro.infer.mcmc")
        is_multiprocessing = log_queue is not None
        progress_bar = None
        if not is_multiprocessing:
            progress_bar = ProgressBar(self.warmup_steps, self.num_samples, disable=self.disable_progbar)
        self.logger = initialize_logger(self.logger, logger_id, progress_bar, log_queue)
        self.kernel.setup(self.warmup_steps, *args, **kwargs)
        params = self.kernel.initial_params
        with optional(progress_bar, not is_multiprocessing):
            for params in self._gen_samples(self.warmup_steps, params):
                continue
            if progress_bar:
                progress_bar.set_description("Sample")
            for params in self._gen_samples(self.num_samples, params):
                trace = self._trace_wrap(params, *args, **kwargs)
                yield (trace, 1.0)
        self.kernel.cleanup()


class MCMC(TracePosterior):
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
    :param str mp_context: Multiprocessing context to use when `num_chains > 1`.
        Only applicable for Python 3.5 and above. Use `mp_context="spawn"` for
        CUDA.
    :param bool disable_progbar: Disable progress bar and diagnostics update.
    """
    def __init__(self, kernel, num_samples, warmup_steps=None,
                 num_chains=1, mp_context=None, disable_progbar=False):
        self.warmup_steps = num_samples if warmup_steps is None else warmup_steps  # Stan
        self.num_samples = num_samples
        if num_chains > 1:
            # verify num_chains is compatible with available CPU.
            available_cpu = max(mp.cpu_count() - 1, 1)  # reserving 1 for the main process.
            if num_chains > available_cpu:
                warnings.warn("num_chains={} is more than available_cpu={}. "
                              "Resetting number of chains to available CPU count."
                              .format(num_chains, available_cpu))
                num_chains = available_cpu
        if num_chains > 1:
            self.sampler = _ParallelSampler(kernel, num_samples, self.warmup_steps,
                                            num_chains, mp_context, disable_progbar)
        else:
            self.sampler = _SingleSampler(kernel, num_samples, self.warmup_steps, disable_progbar)
        super(MCMC, self).__init__(num_chains=num_chains)
        warnings.warn("This interface to MCMC is deprecated and will be removed in the "
                      "next version of Pyro. Please use `pyro.infer.mcmc.api.MCMC` instead.",
                      DeprecationWarning)

    def _traces(self, *args, **kwargs):
        for sample in self.sampler._traces(*args, **kwargs):
            yield sample

    def marginal(self, sites=None):
        """
        Marginalizes latent sites from the sampler.

        :param list sites: optional list of sites for which we need to generate
            the marginal distribution.
        :returns: A :class:`MCMCMarginals` class instance.
        :rtype: :class:`MCMCMarginals`.
        """
        return MCMCMarginals(self, sites)


class MCMCMarginals(Marginals):
    def __init__(self, trace_posterior, sites=None, validate_args=None):
        warnings.warn("This class is deprecated and will be removed in the "
                      "next version of Pyro. Consider using `pyro.infer.util.diagnostics` "
                      "to get diagnostic stats on posterior samples.",
                      DeprecationWarning)
        super(MCMCMarginals, self).__init__(trace_posterior, sites, validate_args)

    def diagnostics(self):
        """
        Gets some diagnostics statistics such as effective sample size and
        split Gelman-Rubin from the sampler.
        """
        if self._diagnostics:
            return self._diagnostics
        support = self.support()
        for site in self.sites:
            site_support = support[site]
            if self._trace_posterior.num_chains == 1:
                site_support = site_support.unsqueeze(0)
            site_stats = OrderedDict()
            try:
                site_stats["n_eff"] = stats.effective_sample_size(site_support)
            except NotImplementedError:
                site_stats["n_eff"] = torch.full(site_support.shape[2:], float("nan"),
                                                 dtype=site_support.dtype, device=site_support.device)
            site_stats["r_hat"] = stats.split_gelman_rubin(site_support)
            self._diagnostics[site] = site_stats
        return self._diagnostics
