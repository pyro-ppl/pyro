# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import sys
from collections import OrderedDict

from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_nb

try:
    get_ipython
    ipython_env = True
except NameError:
    ipython_env = False

# Identifiers to distinguish between diagnostic messages for progress bars
# vs. logging output. Useful when using QueueHandler in multiprocessing.
LOG_MSG = "LOG"
DIAGNOSTIC_MSG = "DIAGNOSTICS"


# Following compatibility code is for Python 2 (available in Python 3.2+).
# Source: https://github.com/python/cpython/blob/master/Lib/logging/handlers.py
#
# Copyright 2001-2016 by Vinay Sajip. All Rights Reserved.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose and without fee is hereby granted,
# provided that the above copyright notice appear in all copies and that
# both that copyright notice and this permission notice appear in
# supporting documentation, and that the name of Vinay Sajip
# not be used in advertising or publicity pertaining to distribution
# of the software without specific, written prior permission.
# VINAY SAJIP DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
# ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# VINAY SAJIP BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR
# ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER
# IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.


class ProgressBar:
    """
    Initialize progress bars using :class:`~tqdm.tqdm`.

    :param int warmup_steps: Number of warmup steps.
    :param int num_samples: Number of MCMC samples.
    :param int min_width: Minimum column width of the bar.
    :param int max_width: Maximum column width of the bar.
    :param bool disable: Disable progress bar.
    :param int num_bars: Number of progress bars to initialize.
        If multiple bars are initialized, they need to be separately
        updated via the ``pos`` kwarg.
    """
    def __init__(self, warmup_steps, num_samples, min_width=80, max_width=120,
                 disable=False, num_bars=1):
        total_steps = warmup_steps + num_samples
        # Disable progress bar in "CI"
        # (see https://github.com/travis-ci/travis-ci/issues/1337).
        disable = disable or "CI" in os.environ or "PYTEST_XDIST_WORKER" in os.environ
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
        pbar_cls = tqdm_nb if num_bars > 1 and ipython_env else tqdm
        self.progress_bars = []
        for i in range(num_bars):
            description = "Warmup" if num_bars == 1 else "Warmup [{}]".format(i + 1)
            pbar = pbar_cls(total=total_steps, desc=description, bar_format=bar_format,
                            position=i, file=sys.stderr, disable=disable)
            # Assume reasonable values when terminal width not available
            if getattr(pbar, "ncols", None) is not None:
                pbar.ncols = max(min_width, pbar.ncols)
                pbar.ncols = min(max_width, pbar.ncols)
            self.progress_bars.append(pbar)
        self.disable = disable
        self.ipython_env = ipython_env

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def set_description(self, *args, **kwargs):
        pos = kwargs.pop("pos", 0)
        if not self.disable:
            self.progress_bars[pos].set_description(*args, **kwargs)

    def set_postfix(self, *args, **kwargs):
        pos = kwargs.pop("pos", 0)
        if not self.disable:
            self.progress_bars[pos].set_postfix(*args, **kwargs)

    def update(self, *args, **kwargs):
        pos = kwargs.pop("pos", 0)
        if not self.disable:
            self.progress_bars[pos].update(*args, **kwargs)

    def close(self):
        for pbar in self.progress_bars:
            pbar.close()
        # Required to not overwrite multiple progress bars on exit.
        if not self.ipython_env and not self.disable:
            sys.stderr.write("\n" * len(self.progress_bars))


class QueueHandler(logging.Handler):
    """
    This handler sends events to a queue. Typically, it would be used together
    with a multiprocessing Queue to centralise logging to file in one process
    (in a multi-process application), so as to avoid file write contention
    between processes.

    This code is new in Python 3.2, but this class can be copy pasted into
    user code for use with earlier Python versions.
    """

    def __init__(self, queue):
        """
        Initialise an instance, using the passed queue.
        """
        logging.Handler.__init__(self)
        self.queue = queue

    def enqueue(self, record):
        """
        Enqueue a record.

        The base implementation uses put_nowait. You may want to override
        this method if you want to use blocking, timeouts or custom queue
        implementations.
        """
        self.queue.put_nowait(record)

    def prepare(self, record):
        """
        Prepares a record for queuing. The object returned by this method is
        enqueued.

        The base implementation formats the record to merge the message
        and arguments, and removes unpickleable items from the record
        in-place.

        You might want to override this method if you want to convert
        the record to a dict or JSON string, or send a modified copy
        of the record while leaving the original intact.
        """
        record.msg = self.format(record)
        record.args = None
        record.exc_info = None
        return record

    def emit(self, record):
        """
        Emit a record.

        Writes the LogRecord to the queue, preparing it for pickling first.
        """
        try:
            self.enqueue(self.prepare(record))
        except Exception:
            self.handleError(record)


class TqdmHandler(logging.StreamHandler):
    """
    Handler that synchronizes the log output with the
    :class:`~tqdm.tqdm` progress bar.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            self.flush()
            tqdm.write(msg, file=sys.stderr)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class MCMCLoggingHandler(logging.Handler):
    """
    Main logging handler used by :class:`~pyro.infer.mcmc`,
    to handle both progress bar updates and regular `logging`
    messages.

    :param log_handler: default log handler for logging
        output.
    :param progress_bar: If provided, diagnostic information
        is updated using the bar.
    """
    def __init__(self, log_handler, progress_bar=None):
        logging.Handler.__init__(self)
        self.log_handler = log_handler
        self.progress_bar = progress_bar

    def emit(self, record):
        try:
            if self.progress_bar and record.msg_type == DIAGNOSTIC_MSG:
                diagnostics = json.loads(record.getMessage(),
                                         object_pairs_hook=OrderedDict)
                self.progress_bar.set_postfix(diagnostics, refresh=False)
                self.progress_bar.update()
            else:
                self.log_handler.handle(record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


class MetadataFilter(logging.Filter):
    """
    Adds auxiliary information to log records, like `logger_id` and
    `msg_type`.
    """
    def __init__(self, logger_id):
        self.logger_id = logger_id
        super().__init__()

    def filter(self, record):
        record.logger_id = self.logger_id
        if not getattr(record, "msg_type", None):
            record.msg_type = LOG_MSG
        return True


def initialize_logger(logger, logger_id, progress_bar=None, log_queue=None):
    """
    Initialize logger for the :class:`pyro.infer.mcmc` module.

    :param logger: logger instance.
    :param str logger_id: identifier for the log record,
        e.g. chain id in case of multiple samplers.
    :param progress_bar: a :class:`tqdm.tqdm` instance.
    """
    # Reset handler with new `progress_bar`.
    logger.handlers = []
    logger.propagate = False
    if log_queue:
        handler = QueueHandler(log_queue)
        format = "[%(levelname)s %(msg_type)s %(logger_id)s]%(message)s"
        progress_bar = None
    elif progress_bar:
        format = "%(levelname).1s \t %(message)s"
        handler = TqdmHandler()
    else:
        raise ValueError("Logger cannot be initialized without a "
                         "valid handler.")
    handler.setFormatter(logging.Formatter(format))
    logging_handler = MCMCLoggingHandler(handler, progress_bar)
    logging_handler.addFilter(MetadataFilter(logger_id))
    logger.addHandler(logging_handler)
    return logger
