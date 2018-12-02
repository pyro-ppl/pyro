import json
import logging
from collections import OrderedDict

# Identifiers to distinguish between diagnostic messages for progress bars
# vs. logging output. Useful when using QueueHandler in multiprocessing.
from pyro.contrib.viz.progress_bar import ProgressBar

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


class MCMCStreamHandler(logging.StreamHandler):
    """
    Handler that synchronizes the log output with the
    :class:`~tqdm.tqdm` progress bar.
    """
    def emit(self, record):
        try:
            msg = self.format(record)
            self.flush()
            ProgressBar.write(msg)
        except (KeyboardInterrupt, SystemExit) as e:
            raise e
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
                self.progress_bar.set_postfix(diagnostics)
                self.progress_bar.increment(update=True)
            else:
                self.log_handler.handle(record)
        except (KeyboardInterrupt, SystemExit) as e:
            raise e
        except Exception:
            self.handleError(record)


class MetadataFilter(logging.Filter):
    """
    Adds auxiliary information to log records, like `logger_id` and
    `msg_type`.
    """
    def __init__(self, logger_id):
        self.logger_id = logger_id
        super(MetadataFilter, self).__init__()

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
    :param progress_bar: a :class:`~pyro.contrib.viz.ProgressBar` instance.
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
        handler = MCMCStreamHandler()
    else:
        raise ValueError("Logger cannot be initialized without a "
                         "valid handler.")
    handler.setFormatter(logging.Formatter(format))
    logging_handler = MCMCLoggingHandler(handler, progress_bar)
    logging_handler.addFilter(MetadataFilter(logger_id))
    logger.addHandler(logging_handler)
    return logger
