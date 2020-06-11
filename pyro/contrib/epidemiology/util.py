# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import numpy
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape
from pyro.ops.special import safe_log


def clamp(tensor, *, min=None, max=None):
    """
    Like :func:`torch.clamp` but dispatches to :func:`torch.min` and/or
    :func:`torch.max` if ``min`` and/or ``max`` is a :class:`~torch.Tensor`.
    """
    if isinstance(min, torch.Tensor):
        tensor = torch.max(tensor, min)
        min = None
    if isinstance(max, torch.Tensor):
        tensor = torch.min(tensor, max)
        max = None
    if min is None and max is None:
        return tensor
    return tensor.clamp(min=min, max=max)


def cat2(lhs, rhs, *, dim=-1):
    """
    Like ``torch.cat([lhs, rhs], dim=dim)`` but dispatches to
    :func:`torch.nn.functional.pad` in case one of ``lhs`` or ``rhs`` is a
    scalar.
    """
    assert dim < 0
    if not isinstance(lhs, torch.Tensor):
        pad = (0, 0) * (-1 - dim) + (1, 0)
        return torch.nn.functional.pad(rhs, pad, value=lhs)
    if not isinstance(rhs, torch.Tensor):
        pad = (0, 0) * (-1 - dim) + (0, 1)
        return torch.nn.functional.pad(lhs, pad, value=rhs)

    diff = lhs.dim() - rhs.dim()
    if diff > 0:
        rhs = rhs.expand((1,) * diff + rhs.shape)
    elif diff < 0:
        diff = -diff
        lhs = lhs.expand((1,) * diff + lhs.shape)
    shape = list(broadcast_shape(lhs.shape, rhs.shape))
    shape[dim] = -1
    return torch.cat([lhs.expand(shape), rhs.expand(shape)], dim=dim)


@torch.no_grad()
def align_samples(samples, model, particle_dim):
    """
    Unsqueeze stacked samples such that their particle dim all aligns.
    This traces ``model`` to determine the ``event_dim`` of each site.
    """
    assert particle_dim < 0

    sample = {name: value[0] for name, value in samples.items()}
    with poutine.block(), poutine.trace() as tr, poutine.condition(data=sample):
        model()

    samples = samples.copy()
    for name, value in samples.items():
        event_dim = tr.trace.nodes[name]["fn"].event_dim
        pad = event_dim - particle_dim - value.dim()
        if pad < 0:
            raise ValueError("Cannot align samples, try moving particle_dim left")
        if pad > 0:
            shape = value.shape[:1] + (1,) * pad + value.shape[1:]
            samples[name] = value.reshape(shape)

    return samples


# this 8 x 10 tensor encodes the coefficients of 8 10-dimensional polynomials
# that are used to construct the num_quant_bins=16 quantization strategy

W16 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1562511562511555e-07],
       [1.1562511562511557e-07, 1.04062604062604e-06, 4.16250416250416e-06,
        9.712509712509707e-06, 1.456876456876456e-05, 1.4568764568764562e-05,
        9.712509712509707e-06, 4.16250416250416e-06, 1.04062604062604e-06, -6.937506937506934e-07],
       [5.839068339068337e-05, 0.0002591158841158841, 0.0005036630036630038,
        0.0005536130536130536, 0.00036421911421911425, 0.00013111888111888106,
        9.712509712509736e-06, -1.2487512487512482e-05, -5.2031302031302014e-06, 1.6187516187516182e-06],
       [0.0018637612387612374, 0.004983558108558107, 0.005457042957042955,
        0.0029234654234654212, 0.000568181818181818, -0.0001602564102564102,
        -8.741258741258739e-05, 4.162504162504162e-06, 9.365634365634364e-06, -1.7536475869809201e-06],
       [0.015560115039281694, 0.025703289765789755, 0.015009296259296255,
        0.0023682336182336166, -0.000963966588966589, -0.00029380341880341857,
        5.6656306656306665e-05, 1.5956265956265953e-05, -6.417193917193917e-06, 7.515632515632516e-07],
       [0.057450111616778265, 0.05790875790875791, 0.014424464424464418,
        -0.0030303030303030303, -0.0013791763791763793, 0.00011655011655011669,
        5.180005180005181e-05, -8.325008325008328e-06, 3.4687534687534703e-07, 0.0],
       [0.12553422657589322, 0.072988122988123, -0.0011641136641136712,
        -0.006617456617456618, -0.00028651903651903725, 0.00027195027195027195,
        3.2375032375032334e-06, -5.550005550005552e-06, 3.4687534687534703e-07, 0.0],
       [0.21761806865973532, 1.7482707128494565e-17, -0.028320290820290833,
        0.0, 0.0014617327117327117, 0.0,
        -3.561253561253564e-05, 0.0, 3.4687534687534714e-07, 0.0]]

W16 = numpy.array(W16)


def compute_bin_probs(s, num_quant_bins):
    """
    Compute categorical probabilities for a quantization scheme with num_quant_bins many
    bins. `s` is a real-valued tensor with values in [0, 1]. Returns probabilities
    of shape `s.shape` + `(num_quant_bins,)`
    """

    t = 1 - s

    if num_quant_bins == 2:
        probs = torch.stack([t, s], dim=-1)
        return probs

    ss = s * s
    tt = t * t

    if num_quant_bins == 4:
        # This cubic spline interpolates over the nearest four integers, ensuring
        # piecewise quadratic gradients.
        probs = torch.stack([
            t * tt,
            4 + ss * (3 * s - 6),
            4 + tt * (3 * t - 6),
            s * ss,
        ], dim=-1) * (1/6)
        return probs

    if num_quant_bins == 8:
        # This quintic spline interpolates over the nearest eight integers, ensuring
        # piecewise quartic gradients.
        s3 = ss * s
        s4 = ss * ss
        s5 = s3 * ss

        t3 = tt * t
        t4 = tt * tt
        t5 = t3 * tt

        probs = torch.stack([
            2 * t5,
            2 + 10 * t + 20 * tt + 20 * t3 + 10 * t4 - 7 * t5,
            55 + 115 * t + 70 * tt - 9 * t3 - 25 * t4 + 7 * t5,
            302 - 100 * ss + 10 * s4,
            302 - 100 * tt + 10 * t4,
            55 + 115 * s + 70 * ss - 9 * s3 - 25 * s4 + 7 * s5,
            2 + 10 * s + 20 * ss + 20 * s3 + 10 * s4 - 7 * s5,
            2 * s5
        ], dim=-1) * (1/840)
        return probs

    if num_quant_bins == 12:
        # This septic spline interpolates over the nearest 12 integers
        s3 = ss * s
        s4 = ss * ss
        s5 = s3 * ss
        s6 = s3 * s3
        s7 = s4 * s3

        t3 = tt * t
        t4 = tt * tt
        t5 = t3 * tt
        t6 = t3 * t3
        t7 = t4 * t3

        probs = torch.stack([
            693 * t7,
            693 + 4851 * t + 14553 * tt + 24255 * t3 + 24255 * t4 + 14553 * t5 + 4851 * t6 - 3267 * t7,
            84744 + 282744 * t + 382536 * tt + 249480 * t3 + 55440 * t4 - 24948 * t5 - 18018 * t6 + 5445 * t7,
            1017423 + 1823283 * t + 1058211 * tt + 51975 * t3 - 148995 * t4 - 18711 * t5 + 20097 * t6 - 3267 * t7,
            3800016 + 3503808 * t + 365904 * tt - 443520 * t3 - 55440 * t4 + 33264 * t5 - 2772 * t6,
            8723088 - 1629936 * ss + 110880.0 * s4 - 2772 * s6,
            8723088 - 1629936 * tt + 110880.0 * t4 - 2772 * t6,
            3800016 + 3503808 * s + 365904 * ss - 443520 * s3 - 55440 * s4 + 33264 * s5 - 2772 * s6,
            1017423 + 1823283 * s + 1058211 * ss + 51975 * s3 - 148995 * s4 - 18711 * s5 + 20097 * s6 - 3267 * s7,
            84744 + 282744 * s + 382536 * ss + 249480 * s3 + 55440 * s4 - 24948 * s5 - 18018 * s6 + 5445 * s7,
            693 + 4851 * s + 14553 * ss + 24255 * s3 + 24255 * s4 + 14553 * s5 + 4851 * s6 - 3267 * s7,
            693 * s7,
        ], dim=-1) * (1/32931360)
        return probs

    if num_quant_bins == 16:
        # This nonic spline interpolates over the nearest 16 integers
        w16 = torch.from_numpy(W16).to(s.device).type_as(s)
        s_powers = s.unsqueeze(-1).unsqueeze(-1).pow(torch.arange(10.))
        t_powers = t.unsqueeze(-1).unsqueeze(-1).pow(torch.arange(10.))
        splines_t = (w16 * t_powers).sum(-1)
        splines_s = (w16 * s_powers).sum(-1)
        index = [0, 1, 2, 3, 4, 5, 6, 15, 7, 14, 13, 12, 11, 10, 9, 8]
        probs = torch.cat([splines_t, splines_s], dim=-1)
        probs = probs.index_select(-1, torch.tensor(index))
        return probs

    raise ValueError("Unsupported num_quant_bins: {}".format(num_quant_bins))


def _all(x):
    return x.all() if isinstance(x, torch.Tensor) else x


def _unsqueeze(x):
    return x.unsqueeze(-1) if isinstance(x, torch.Tensor) else x


def quantize(name, x_real, min, max, num_quant_bins=4):
    """Randomly quantize in a way that preserves probability mass."""
    assert _all(min < max)
    if num_quant_bins == 1:
        x = x_real.detach().round()
        return pyro.deterministic(name, x, event_dim=0)

    lb = x_real.detach().floor()

    probs = compute_bin_probs(x_real - lb, num_quant_bins=num_quant_bins)

    q = pyro.sample("Q_" + name, dist.Categorical(probs),
                    infer={"enumerate": "parallel"})
    q = q.type_as(x_real) - (num_quant_bins // 2 - 1)

    x = lb + q
    x = torch.max(x, 2 * min - 1 - x)
    x = torch.min(x, 2 * max + 1 - x)

    return pyro.deterministic(name, x, event_dim=0)


def quantize_enumerate(x_real, min, max, num_quant_bins=4):
    """Quantize, then manually enumerate."""
    assert _all(min < max)
    lb = x_real.detach().floor()

    probs = compute_bin_probs(x_real - lb, num_quant_bins=num_quant_bins)
    logits = safe_log(probs)

    arange_min = 1 - num_quant_bins // 2
    arange_max = 1 + num_quant_bins // 2
    q = torch.arange(arange_min, arange_max)

    x = lb.unsqueeze(-1) + q
    x = torch.max(x, 2 * _unsqueeze(min) - 1 - x)
    x = torch.min(x, 2 * _unsqueeze(max) + 1 - x)

    return x, logits
