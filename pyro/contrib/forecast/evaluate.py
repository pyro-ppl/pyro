# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging

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


def backtest(data, covariates, model, *,
             metrics=None,
             transform=None,
             train_window=None,
             min_train_window=1,
             test_window=None,
             min_test_window=1,
             stride=1,
             seed=1234567890,
             num_samples=100,
             forecaster_options={}):
    """
    Backtest a forecasting model on a moving window of (train,test) data.

    :param data: A tensor dataset with time dimension -2.
    :type data: ~torch.Tensor
    :param covariates: A tensor of covariates with time dimension -2.
        For models not using covariates, pass a shaped empty tensor
        ``torch.empty(duration, 0)``.
    :type covariates: ~torch.Tensor
    :param model:
    :type model: ~pyro.contrib.forecast.forecaster.ForecastingModel

    :param dict metrics: A dictionary mapping metric name to metric function.
        The metric function should input a forecast ``pred`` and ground
        ``truth`` and can output anything, often a number. Example metrics
        include: :func:`eval_mae`, :func:`eval_rmse`, and :func:`eval_crps`.
    :param callable transform: An optional transform to apply before computing
        metrics. If provided this will be applied as
        ``pred, truth = transform(pred, truth)``.
    :param int train_window: Size of the training window. Be default trains
        from beginning of data.
    :param int min_train_window: If ``train_window`` is None, this specifies
        the min training window size. Defaults to 1.
    :param int test_window: Size of the test window. By default forecasts to
        end of data.
    :param int min_test_window: If ``test_window`` is None, this specifies
        the min test window size. Defaults to 1.
    :param int stride: Optional stride for test/train split. Defaults to 1.
    :param int seed: Random number seed.
    :param int num_samples: Number of samples for forecast.
    :param dict forecaster_options: Options to pass to forecaster. See
        :class:`~pyro.contrib.forecaster.Forecaster` for details.

    :returns: A list of dictionaries of evaluation data. Caller is responsible
        for aggregating the per-window metrics. Dictionary keys include: train
        begin time "t0", train/test split time "t1", test end  time "t2",
        "seed", "num_samples" and one key for each metric.
    :rtype: list
    """
    assert data.size(-2) == covariates.size(-2)
    assert isinstance(min_train_window, int) and min_train_window >= 1
    assert isinstance(min_test_window, int) and min_test_window >= 1
    if metrics is None:
        metrics = DEFAULT_METRICS
    assert metrics, "no metrics specified"

    duration = data.size(-2)
    if test_window is None:
        stop = duration - min_test_window + 1
    else:
        stop = duration - test_window + 1
    if train_window is None:
        start = min_train_window
    else:
        start = train_window

    results = []
    for t1 in range(start, stop, stride):
        t0 = 0 if train_window is None else t1 - train_window
        t2 = duration if test_window is None else t1 + test_window
        assert 0 <= t0 < t1 < t2 <= duration
        logger.info("Evaluating on window ({}, {}, {})".format(t0, t1, t2))

        # Train a forecaster on the training window.
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        train_data = data[..., t0:t1, :]
        train_covariates = covariates[..., t0:t1, :]
        forecaster = Forecaster(model, train_data, train_covariates, **forecaster_options)

        # Forecast forward to testing window.
        test_covariates = covariates[..., t0:t2, :]
        pred = forecaster(train_data, test_covariates, num_samples=num_samples)
        truth = data[..., t1:t2, :]

        # We aggressively garbage collect because Monte Carlo forecast are memory intensive.
        pyro.clear_param_store()
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
        }
        results.append(result)
        for name, fn in metrics.items():
            result[name] = fn(pred, truth)
            logger.debug("{} = {}".format(name, result[name]))

        del pred

    return results
