# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import inspect
import io

import pytest
import pickle
import torch

import pyro.distributions as dist
from pyro.distributions.torch_distribution import TorchDistributionMixin
from tests.common import xfail_param

# Collect distributions.
BLACKLIST = [
    dist.TorchDistribution,
    dist.ExponentialFamily,
    dist.OMTMultivariateNormal,
]
XFAIL = {
    dist.Gumbel: xfail_param(dist.Gumbel, reason='cannot pickle weakref'),
}
DISTRIBUTIONS = [d for d in dist.__dict__.values()
                 if isinstance(d, type)
                 if issubclass(d, TorchDistributionMixin)
                 if d not in BLACKLIST]
DISTRIBUTIONS.sort(key=lambda d: d.__name__)
DISTRIBUTIONS = [XFAIL.get(d, d) for d in DISTRIBUTIONS]

# Provide default args if Dist(1, 1, ..., 1) is known to fail.
ARGS = {
    dist.AVFMultivariateNormal: [torch.zeros(3), torch.eye(3), torch.rand(2, 4, 3)],
    dist.Bernoulli: [0.5],
    dist.Binomial: [2, 0.5],
    dist.Categorical: [torch.ones(2)],
    dist.Delta: [torch.tensor(0.)],
    dist.Dirichlet: [torch.ones(2)],
    dist.GaussianScaleMixture: [torch.ones(2), torch.ones(3), torch.ones(3)],
    dist.Geometric: [0.5],
    dist.Independent: [dist.Normal(torch.zeros(2), torch.ones(2)), 1],
    dist.LowRankMultivariateNormal: [torch.zeros(2), torch.ones(2, 2), torch.ones(2)],
    dist.MaskedMixture: [torch.tensor([1, 0]).bool(), dist.Normal(0, 1), dist.Normal(0, 2)],
    dist.MixtureOfDiagNormals: [torch.ones(2, 3), torch.ones(2, 3), torch.ones(2)],
    dist.MixtureOfDiagNormalsSharedCovariance: [torch.ones(2, 3), torch.ones(3), torch.ones(2)],
    dist.Multinomial: [2, torch.ones(2)],
    dist.MultivariateNormal: [torch.ones(2), torch.eye(2)],
    dist.OneHotCategorical: [torch.ones(2)],
    dist.RelaxedBernoulli: [1.0, 0.5],
    dist.RelaxedBernoulliStraightThrough: [1.0, 0.5],
    dist.RelaxedOneHotCategorical: [1., torch.ones(2)],
    dist.RelaxedOneHotCategoricalStraightThrough: [1., torch.ones(2)],
    dist.TransformedDistribution: [dist.Normal(0, 1), torch.distributions.ExpTransform()],
    dist.Uniform: [0, 1],
    dist.VonMises3D: [torch.tensor([1., 0., 0.])],
}


@pytest.mark.parametrize('Dist', DISTRIBUTIONS)
def test_pickle(Dist):
    if Dist in ARGS:
        args = ARGS[Dist]
    else:
        # Optimistically try to initialize with Dist(1, 1, ..., 1).
        try:
            # Python 3.6+
            spec = list(inspect.signature(Dist.__init__).parameters.values())
            nargs = sum(1 for p in spec if p.default is p.empty) - 1
        except AttributeError:
            # Python 2.6-3.5
            spec = inspect.getargspec(Dist.__init__)
            nargs = len(spec.args) - 1 - (len(spec.defaults) if spec.defaults else 0)
        args = (1,) * nargs
    try:
        dist = Dist(*args)
    except Exception:
        pytest.skip(msg='cannot construct distribution')

    buffer = io.BytesIO()
    # Note that pickling torch.Size() requires protocol >= 2
    torch.save(dist, buffer, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    buffer.seek(0)
    deserialized = torch.load(buffer)
    assert isinstance(deserialized, Dist)
