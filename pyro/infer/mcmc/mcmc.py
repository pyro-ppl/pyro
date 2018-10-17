from __future__ import absolute_import, division, print_function

import errno
import json
import logging
import signal
import socket
import sys
import threading
from collections import OrderedDict

import six
from six.moves import queue
import torch
import torch.multiprocessing as mp

import pyro
from pyro.infer import TracePosterior
from pyro.infer.mcmc.logger import initialize_logger, initialize_progbar, TQDM_MSG, TqdmHandler
from pyro.util import optional


def logger_thread(queue, warmup_steps, num_samples, num_chains):
    progress_bars = [initialize_progbar(warmup_steps, s, pos=i)
                     for i, s in enumerate(num_samples)]
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.addHandler(TqdmHandler())
    num_samples = [0] * len(num_samples)
    while True:
        record = queue.get()
        if record is None:
            break
        metadata, msg = record.getMessage().split("]", 1)
        _, msg_type, _, chain_id = metadata[1:].split()
        chain_id = int(chain_id)
        if msg_type == TQDM_MSG:
            num_samples[chain_id] += 1
            if num_samples[chain_id] == warmup_steps:
                progress_bars[chain_id].set_description("Sample [{}]".format(chain_id))
            diagnostics = json.loads(msg, object_pairs_hook=OrderedDict)
            progress_bars[chain_id].set_postfix(diagnostics)
            progress_bars[chain_id].update()
        else:
            logger.handle(record)
    for pbar in progress_bars:
        pbar.close()
    sys.stderr.write("\n" * num_chains)


class _Worker(object):
    def __init__(self, chain_id, result_queue, log_queue,
                 kernel, num_samples, warmup_steps=0,
                 args=None, kwargs=None):
        self.chain_id = chain_id
        self.trace_gen = _SingleSampler(kernel, num_samples=num_samples, warmup_steps=warmup_steps)
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.rng_seed = torch.initial_seed()
        self.log_queue = log_queue
        self.result_queue = result_queue
        self.default_tensor_type = torch.Tensor().type()

    def run(self, *args, **kwargs):
        pyro.set_rng_seed(self.chain_id + self.rng_seed)
        torch.set_default_tensor_type(self.default_tensor_type)
        kwargs["chain_id"] = self.chain_id
        kwargs["log_queue"] = self.log_queue
        try:
            for sample in self.trace_gen._traces(*args, **kwargs):
                self.result_queue.put_nowait((self.chain_id, sample))
            self.result_queue.put_nowait((self.chain_id, None))
        except Exception as e:
            self.trace_gen.logger.exception(e)
            self.result_queue.put_nowait((self.chain_id, e))


class _ParallelSampler(TracePosterior):
    """
    Parallel runner when MCMC is run with multiple chains.
    """
    def __init__(self, kernel, num_samples, warmup_steps, num_chains, mp_context):
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
        self.result_queue = self.ctx.Manager().Queue()
        self.log_queue = self.ctx.Manager().Queue()
        # initialize number of samples per chain
        samples_per_chain = num_samples // num_chains
        self.num_samples = [samples_per_chain] * num_chains
        for i in range(num_samples % num_chains):
            self.num_samples[i] += 1
        self.log_thread = threading.Thread(target=logger_thread,
                                           args=(self.log_queue, self.warmup_steps,
                                                 self.num_samples, self.num_chains))
        self.log_thread.daemon = True
        self.log_thread.start()

    def init_workers(self, *args, **kwargs):
        self.workers = []
        for i in range(self.num_chains):
            worker = _Worker(i, self.result_queue, self.log_queue, self.kernel,
                             self.num_samples[i], self.warmup_steps)
            worker.daemon = True
            self.workers.append(self.ctx.Process(name=str(i), target=worker.run,
                                                 args=args, kwargs=kwargs))

    def join(self):
        for w in self.workers:
            if w.is_alive():
                w.join(timeout=1)
        if self.log_thread.is_alive():
            self.log_queue.put_nowait(None)
            self.log_thread.join(timeout=1)

    def _traces(self, *args, **kwargs):
        sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.init_workers(*args, **kwargs)
        signal.signal(signal.SIGINT, sigint_handler)
        active_workers = self.num_chains
        try:
            for w in self.workers:
                w.start()
            while active_workers:
                try:
                    chain_id, val = self.result_queue.get(block=False)
                # This can happen when the worker process has terminated.
                # See https://github.com/pytorch/pytorch/pull/5380 for motivation.
                except socket.error as e:
                    if getattr(e, "errno", None) == errno.ENOENT:
                        pass
                    else:
                        raise e
                except queue.Empty:
                    continue
                if isinstance(val, Exception):
                    raise Exception("Exception in chain {}.".format(chain_id))
                elif val is not None:
                    yield val
                else:
                    active_workers -= 1
        finally:
            self.join()


class _SingleSampler(TracePosterior):
    """
    Single runner instance optimized for the case `num_chains=1`.
    """
    def __init__(self, kernel, num_samples, warmup_steps):
        self.kernel = kernel
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.logger = None
        super(_SingleSampler, self).__init__()

    def _gen_samples(self, num_samples, init_trace):
        trace = init_trace
        for _ in range(num_samples):
            trace = self.kernel.sample(trace)
            diagnostics = json.dumps(self.kernel.diagnostics())
            self.logger.info(diagnostics, extra={"msg_type": TQDM_MSG})
            yield trace

    def _traces(self, *args, **kwargs):
        chain_id = kwargs.pop("chain_id", 0)
        log_queue = kwargs.pop("log_queue", None)
        self.logger = logging.getLogger("pyro.infer.mcmc")
        is_multiprocessing = log_queue is not None
        progress_bar = initialize_progbar(self.warmup_steps, self.num_samples) \
            if not is_multiprocessing else None
        self.logger = initialize_logger(self.logger, chain_id, progress_bar, log_queue)
        self.kernel.setup(self.warmup_steps, *args, **kwargs)
        trace = self.kernel.initial_trace()
        with optional(progress_bar, not is_multiprocessing):
            for trace in self._gen_samples(self.warmup_steps, trace):
                continue
            if progress_bar:
                progress_bar.set_description("Sample")
            for trace in self._gen_samples(self.num_samples, trace):
                yield (trace, 1.0)
        self.kernel.cleanup()


class MCMC(TracePosterior):
    """
    Wrapper class for Markov Chain Monte Carlo algorithms. Specific MCMC algorithms
    are TraceKernel instances and need to be supplied as a ``kernel`` argument
    to the constructor.

    :param kernel: An instance of the ``TraceKernel`` class, which when
        given an execution trace returns another sample trace from the target
        (posterior) distribution.
    :param int num_samples: The number of samples that need to be generated,
        excluding the samples discarded during the warmup phase.
    :param int warmup_steps: Number of warmup iterations. The samples generated
        during the warmup phase are discarded. If not provided, default is
        half of `num_samples`.
    :param int num_chain: Number of MCMC chains to run in parallel.
    :param str mp_context: Multiprocessing context to use when `num_chains > 1`.
        Only applicable for Python 3.5 and above.
    """
    def __init__(self, kernel, num_samples, warmup_steps=0,
                 num_chains=1, mp_context=None):
        self.warmup_steps = warmup_steps if warmup_steps is not None else num_samples // 2  # Stan
        self.num_samples = num_samples
        if num_chains > 1:
            self.sampler = _ParallelSampler(kernel, num_samples, warmup_steps,
                                            num_chains, mp_context)
        else:
            self.sampler = _SingleSampler(kernel, num_samples, warmup_steps)
        super(MCMC, self).__init__()

    def _traces(self, *args, **kwargs):
        for sample in self.sampler._traces(*args, **kwargs):
            yield sample
