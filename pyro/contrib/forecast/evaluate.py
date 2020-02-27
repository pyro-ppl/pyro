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


def backtest(data, covariates, model_fn, guide_fn=None, *,
             metrics=None,
             transform=None,
             train_window=None,
             min_train_window=1,
             test_window=None,
             min_test_window=1,
             stride=1,
             seed=1234567890,
             num_samples=100,
             warm_start=False,
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
        ~pyro.contrib.forecast.forecaster.ForecastingModel object.
    :param callable guide_fn: Function that returns a guide object.

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
    :param bool warm_start: Whether to warm start parameters from the previous
        time window. Note this may introduce statistical leakage; it is
        recommended for model expoloration purposes only and should be disabled
        when publishing metrics.
    :param forecaster_options: Options dict to pass to forecaster, or callable
        inputting time window ``t0,t1,t2`` and returning such a dict. See
        :class:`~pyro.contrib.forecaster.Forecaster` for details.
    :type forecaster_options: dict or callable

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
    forecaster_options_fn = None
    if callable(forecaster_options):
        forecaster_options_fn = forecaster_options

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
        if warm_start and results:
            _warm_start_param_store(t0=t0, t1=t1, t0_old=results[-1]["t0"],
                                    t1_old=results[-1]["t1"])
        train_data = data[..., t0:t1, :]
        train_covariates = covariates[..., t0:t1, :]
        model = model_fn()
        guide = None if guide_fn is None else guide_fn()
        if forecaster_options_fn is not None:
            forecaster_options = forecaster_options_fn(t0=t0, t1=t1, t2=t2)
        forecaster = Forecaster(model, train_data, train_covariates,
                                guide=guide, **forecaster_options)

        # Forecast forward to testing window.
        test_covariates = covariates[..., t0:t2, :]
        pred = forecaster(train_data, test_covariates, num_samples=num_samples)
        truth = data[..., t1:t2, :]

        # We aggressively garbage collect because Monte Carlo forecast are memory intensive.
        del forecaster
        if not warm_start:
            pyro.clear_param_store()

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


def _warm_start_param_store(t0, t1, t0_old, t1_old):
    """
    Helper to update params in the param store in-place as the backtesting
    window changes. New blocks are initialized to zero in unconstrained space,
    equivalent to :func:`pyro.infer.autoguide.initialization.init_to_feasible`.
    """
    assert t0_old <= t0
    assert t1_old <= t1
    assert (t0_old < t0) or (t1_old < t1)
    store = pyro.get_param_store()
    for name, constrained_param in store.items():
        param = constrained_param.unconstrained()
        time_dim = getattr(param, "_pyro_time_dim", None)
        if time_dim is None:
            continue
        assert time_dim < 0
        time_dim += constrained_param.dim() - param.dim()
        with torch.no_grad():

            # Truncate params that have just left the window.
            if t0 != t0_old:
                dots = (slice(None),) * (param.dim() + time_dim)
                param = param[dots + (slice(t0_old - t0, None),)]

            # Pad zeros for data that has just entered the window.
            padding = t1 - t0 - param.size(time_dim)
            if padding:
                param = torch.nn.functional.pad(param, (0, 0) * (-1 - time_dim) + (0, padding))

        # Set the unconstrained param.
        param.requires_grad_(True)
        store._params[name] = param
        store._param_to_name[param] = name
