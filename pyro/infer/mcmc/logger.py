import json
import logging
import os
import sys
from collections import OrderedDict

from tqdm.autonotebook import tqdm

LOG_MSG = "LOG"
TQDM_MSG = "TQDM"


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
            if self.progress_bar and record.msg_type == TQDM_MSG:
                diagnostics = json.loads(record.getMessage(),
                                         object_pairs_hook=OrderedDict)
                self.progress_bar.set_postfix(diagnostics)
                self.progress_bar.update()
            else:
                self.log_handler.handle(record)
        except (KeyboardInterrupt, SystemExit) as e:
            raise e
        except Exception:
            self.handleError(record)


class MetadataFilter(logging.Filter):
    """
    Adds auxiliary information to log records, like `chain_id` and
    `msg_type`.
    """
    def __init__(self, chain_id):
        self.chain_id = chain_id
        super(MetadataFilter, self).__init__()

    def filter(self, record):
        record.chain_id = self.chain_id
        if not getattr(record, "msg_type", None):
            record.msg_type = LOG_MSG
        return True


def initialize_progbar(warmup_steps, num_samples, min_width=100, max_width=120, pos=None):
    """
    Initialize progress bar using :class:`~tqdm.tqdm`.

    :param int warmup_steps: Number of warmup steps.
    :param int num_samples: Number of MCMC samples.
    :param int min_width: Minimum column width of the bar.
    :param int max_width: Maximum column width of the bar.
    :param int pos: Position of the bar (e.g. in the case of
        multiple parallel samplers).
    """
    description = "Warmup" if pos is None else "Warmup [{}]".format(pos)
    total_steps = warmup_steps + num_samples
    # Disable progress bar in "CI"
    # (see https://github.com/travis-ci/travis-ci/issues/1337).
    disable = "CI" in os.environ
    progress_bar = tqdm(total=total_steps, desc=description,
                        position=pos, file=sys.stderr, disable=disable)

    if getattr(progress_bar, "ncols", None) is not None:
        progress_bar.ncols = min(min_width, progress_bar.ncols)
        progress_bar.ncols = max(max_width, progress_bar.ncols)
    return progress_bar


def initialize_logger(logger, chain_id, progress_bar=None):
    """
    Initialize logger for the :class:`pyro.infer.mcmc` module.

    :param logger: logger instance.
    :param int chain_id: `id` of the sampler, in case of
        multiple samplers.
    :param progress_bar: a :class:`tqdm.tqdm` instance.
    """
    # Reset handler with new `progress_bar`.
    logger.handlers = []
    logger.propagate = False
    handler = TqdmHandler()
    logging_handler = MCMCLoggingHandler(handler, progress_bar)
    logging_handler.addFilter(MetadataFilter(chain_id))
    logger.addHandler(logging_handler)
    return logger
