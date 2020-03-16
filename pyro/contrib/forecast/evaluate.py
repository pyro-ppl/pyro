# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from timeit import default_timer

import torch

import pyro
from pyro.ops.stats import crps_empirical

from .forecaster import Forecaster

logger = logging.getLogger(__name__)


@torch.no_grad()
def eval_mae(pred, truth):
    """
    Evaluate mean absolute error, using sample median as point estimate.

    :param torch.Tensor pred: Forecasted samples.
    :param torch.Tensor truth: Ground truth.
    :rtype: float
    """
    pred = pred.median(0).values
    return (pred - truth).abs().mean().cpu().item()


@torch.no_grad()
def eval_rmse(pred, truth):
    """
    Evaluate root mean squared error, using sample mean as point estimate.

    :param torch.Tensor pred: Forecasted samples.
    :param torch.Tensor truth: Ground truth.
    :rtype: float
    """
    pred = pred.mean(0)
    error = pred - truth
    return (error * error).mean().cpu().item() ** 0.5


@torch.no_grad()
def eval_crps(pred, truth):
    """
    Evaluate continuous ranked probability score, averaged over all data
    elements.

    **References**

    [1] Tilmann Gneiting, Adrian E. Raftery (2007)
        `Strictly Proper Scoring Rules, Prediction, and Estimation`
        https://www.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf

    :param torch.Tensor pred: Forecasted samples.
    :param torch.Tensor truth: Ground truth.
    :rtype: float
    """
    return crps_empirical(pred, truth).mean().cpu().item()


DEFAULT_METRICS = {
    "mae": eval_mae,
    "rmse": eval_rmse,
    "crps": eval_crps,
}


def backtest(data, covariates, model_fn, *,
             forecaster_fn=Forecaster,
             metrics=None,
             transform=None,
             train_window=None,
             min_train_window=1,
             test_window=None,
             min_test_window=1,
             stride=1,
             seed=1234567890,
             num_samples=100,
             batch_size=None,
             forecaster_options={}):
    """
    Backtest a forecasting model on a moving window of (train,test) data.

    :param data: A tensor dataset with time dimension -2.
    :type data: ~torch.Tensor
    :param covariates: A tensor of covariates with time dimension -2.
        For models not using covariates, pass a shaped empty tensor
        ``torch.empty(duration, 0)``.
    :type covariates: ~torch.Tensor
    :param callable model_fn: Function that returns an
        :class:`~pyro.contrib.forecast.forecaster.ForecastingModel` object.
    :param callable forecaster_fn: Function that returns a forecaster object
        (for example, :class:`~pyro.contrib.forecast.forecaster.Forecaster`
        or :class:`~pyro.contrib.forecast.forecaster.HMCForecaster`)
        given arguments model, training data, training covariates and
        keyword arguments defined in `forecaster_options`.
    :param dict metrics: A dictionary mapping metric name to metric function.
        The metric function should input a forecast ``pred`` and ground
        ``truth`` and can output anything, often a number. Example metrics
        include: :func:`eval_mae`, :func:`eval_rmse`, and :func:`eval_crps`.
    :param callable transform: An optional transform to apply before computing
        metrics. If provided this will be applied as
        ``pred, truth = transform(pred, truth)``.
    :param int train_window: Size of the training window. Be default trains
        from beginning of data. This must be None if forecaster is
        :class:`~pyro.contrib.forecast.forecaster.Forecaster` and
        ``forecaster_options["warm_start"]`` is true.
    :param int min_train_window: If ``train_window`` is None, this specifies
        the min training window size. Defaults to 1.
    :param int test_window: Size of the test window. By default forecasts to
        end of data.
    :param int min_test_window: If ``test_window`` is None, this specifies
        the min test window size. Defaults to 1.
    :param int stride: Optional stride for test/train split. Defaults to 1.
    :param int seed: Random number seed.
    :param int num_samples: Number of samples for forecast. Defaults to 100.
    :param int batch_size: Batch size for forecast sampling. Defaults to
        ``num_samples``.
    :param forecaster_options: Options dict to pass to forecaster, or callable
        inputting time window ``t0,t1,t2`` and returning such a dict. See
        :class:`~pyro.contrib.forecaster.Forecaster` for details.
    :type forecaster_options: dict or callable

    :returns: A list of dictionaries of evaluation data. Caller is responsible
        for aggregating the per-window metrics. Dictionary keys include: train
        begin time "t0", train/test split time "t1", test end  time "t2",
        "seed", "num_samples", "train_walltime", "test_walltime", and one key
        for each metric.
    :rtype: list
    """
    assert data.size(-2) == covariates.size(-2)
    assert isinstance(min_train_window, int) and min_train_window >= 1
    assert isinstance(min_test_window, int) and min_test_window >= 1
    if metrics is None:
        metrics = DEFAULT_METRICS
    assert metrics, "no metrics specified"

    if callable(forecaster_options):
        forecaster_options_fn = forecaster_options
    else:
        def forecaster_options_fn(*args, **kwargs):
            return forecaster_options
    if train_window is not None and forecaster_options_fn().get("warm_start"):
        raise ValueError("Cannot warm start with moving training window; "
                         "either set warm_start=False or train_window=None")

    duration = data.size(-2)
    if test_window is None:
        stop = duration - min_test_window + 1
    else:
        stop = duration - test_window + 1
    if train_window is None:
        start = min_train_window
    else:
        start = train_window

    pyro.clear_param_store()
    results = []
    for t1 in range(start, stop, stride):
        t0 = 0 if train_window is None else t1 - train_window
        t2 = duration if test_window is None else t1 + test_window
        assert 0 <= t0 < t1 < t2 <= duration
        logger.info("Training on window [{t0}:{t1}], testing on window [{t1}:{t2}]"
                    .format(t0=t0, t1=t1, t2=t2))

        # Train a forecaster on the training window.
        pyro.set_rng_seed(seed)
        forecaster_options = forecaster_options_fn(t0=t0, t1=t1, t2=t2)
        if not forecaster_options.get("warm_start"):
            pyro.clear_param_store()
        train_data = data[..., t0:t1, :]
        train_covariates = covariates[..., t0:t1, :]
        start_time = default_timer()
        model = model_fn()
        forecaster = forecaster_fn(model, train_data, train_covariates,
                                   **forecaster_options)
        train_walltime = default_timer() - start_time

        # Forecast forward to testing window.
        test_covariates = covariates[..., t0:t2, :]
        start_time = default_timer()
        # Gradually reduce batch_size to avoid OOM errors.
        while True:
            try:
                pred = forecaster(train_data, test_covariates, num_samples=num_samples,
                                  batch_size=batch_size)
                break
            except RuntimeError as e:
                if "out of memory" in str(e) and batch_size > 1:
                    batch_size = (batch_size + 1) // 2
                    warnings.warn("out of memory, decreasing batch_size to {}"
                                  .format(batch_size), RuntimeWarning)
                else:
                    raise
        test_walltime = default_timer() - start_time
        truth = data[..., t1:t2, :]

        # We aggressively garbage collect because Monte Carlo forecast are memory intensive.
        del forecaster

        # Evaluate the forecasts.
        if transform is not None:
            pred, truth = transform(pred, truth)
        result = {
            "t0": t0,
            "t1": t1,
            "t2": t2,
            "seed": seed,
            "num_samples": num_samples,
            "train_walltime": train_walltime,
            "test_walltime": test_walltime,
            "params": {},
        }
        results.append(result)
        for name, fn in metrics.items():
            result[name] = fn(pred, truth)
        for name, value in pyro.get_param_store().items():
            if value.numel() == 1:
                value = value.cpu().item()
                result["params"][name] = value
        for dct in (result, result["params"]):
            for key, value in sorted(dct.items()):
                if isinstance(value, (int, float)):
                    logger.debug("{} = {:0.6g}".format(key, value))

        del pred

    return results
