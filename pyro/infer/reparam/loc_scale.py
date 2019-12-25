import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.distributions.util import is_identically_one, is_validation_enabled


class LocScaleReparam:
    """
    Generic decentering reparameterizer for distributions that are specified by
    parameters ``loc`` and ``scale`` (and possibly additional
    ``shape_params``). This can be combined with :func:`pyro.param` to learn a
    centering transform::

        x_centered = pyro.param("x_centered", 0.5,
                                constraint=constraints.unit_interval)
        pyro.sample("x", dist.StudentT(df, loc, scale),
                    infer={"reparam": LocScaleReparameterizer(x_centered,
                                                              shape_params=["df"])})

    [1] Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)
        "Automatic Reparameterisation of Probabilistic Programs"
        https://arxiv.org/pdf/1906.03028.pdf

    :param float centered: optional centered parameter. If None (default) learn
        a per-site centering parameter in ``[0,1]``. If 0, fully decenter the
        distribution; if 1, preserve the centered distribution unchanged.
    :param shape_params: list of additional parameter names to copy unchanged from
        the centered to decentered distribution.
    :type shape_params: tuple or list
    """
    def __init__(self, centered=None, shape_params=()):
        assert centered is None or isinstance(centered, (float, torch.Tensor))
        assert isinstance(shape_params, (tuple, list))
        assert all(isinstance(name, str) for name in shape_params)
        if is_validation_enabled():
            if isinstance(centered, float):
                assert 0 <= centered and centered <= 1
            elif isinstance(centered, torch.Tensor):
                assert (0 <= centered).all()
                assert (centered <= 1).all()
            else:
                assert centered is None
        self.centered = centered
        self.shape_params = shape_params

    def __call__(self, name, fn, obs):
        assert obs is None
        centered = self.centered
        if is_identically_one(centered):
            return name, fn, obs

        # Apply a partial decentering transform.
        params = {key: getattr(fn, key) for key in self.shape_params}
        if self.centered is None:
            centered = pyro.param("{}_centered",
                                  lambda: torch.full_like(fn.loc, 0.5),
                                  constraint=constraints.unit_interval)
        params["loc"] = fn.loc * centered
        params["scale"] = fn.scale ** centered
        decentered_fn = type(fn)(**params)

        # Draw decentered noise.
        decentered_value = pyro.sample("{}_decentered".format(name), decentered_fn)

        # Differentiably transform.
        delta = decentered_value - centered * fn.loc
        value = fn.loc + fn.scale.pow(1 - centered) * delta

        # Simulate a pyro.deterministic() site.
        new_fn = dist.Delta(value, event_dim=fn.event_dim).mask(False)
        return new_fn, value
