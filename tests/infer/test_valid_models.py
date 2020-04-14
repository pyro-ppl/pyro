# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging
import warnings
from collections import defaultdict

import pytest
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.testing import fakes
from pyro.infer import (SVI, EnergyDistance, Trace_ELBO, TraceEnum_ELBO, TraceGraph_ELBO, TraceMeanField_ELBO,
                        TraceTailAdaptive_ELBO, config_enumerate)
from pyro.infer.reparam import LatentStableReparam
from pyro.infer.tracetmc_elbo import TraceTMC_ELBO
from pyro.infer.util import torch_item
from pyro.ops.indexing import Vindex
from pyro.optim import Adam
from tests.common import assert_close

logger = logging.getLogger(__name__)

# This file tests a variety of model,guide pairs with valid and invalid structure.


def EnergyDistance_prior(**kwargs):
    kwargs["prior_scale"] = 0.0
    kwargs.pop("strict_enumeration_warning", None)
    return EnergyDistance(**kwargs)


def EnergyDistance_noprior(**kwargs):
    kwargs["prior_scale"] = 1.0
    kwargs.pop("strict_enumeration_warning", None)
    return EnergyDistance(**kwargs)


def assert_ok(model, guide, elbo, **kwargs):
    """
    Assert that inference works without warnings or errors.
    """
    pyro.clear_param_store()
    inference = SVI(model, guide, Adam({"lr": 1e-6}), elbo)
    inference.step(**kwargs)
    try:
        pyro.set_rng_seed(0)
        loss = elbo.loss(model, guide, **kwargs)
        if hasattr(elbo, "differentiable_loss"):
            try:
                pyro.set_rng_seed(0)
                differentiable_loss = torch_item(elbo.differentiable_loss(model, guide, **kwargs))
            except ValueError:
                pass  # Ignore cases where elbo cannot be differentiated
            else:
                assert_close(differentiable_loss, loss, atol=0.01)
        if hasattr(elbo, "loss_and_grads"):
            pyro.set_rng_seed(0)
            loss_and_grads = elbo.loss_and_grads(model, guide, **kwargs)
            assert_close(loss_and_grads, loss, atol=0.01)
    except NotImplementedError:
        pass  # Ignore cases where loss isn't implemented, eg. TraceTailAdaptive_ELBO


def assert_error(model, guide, elbo, match=None):
    """
    Assert that inference fails with an error.
    """
    pyro.clear_param_store()
    inference = SVI(model,  guide, Adam({"lr": 1e-6}), elbo)
    with pytest.raises((NotImplementedError, UserWarning, KeyError, ValueError, RuntimeError),
                       match=match):
        inference.step()


def assert_warning(model, guide, elbo):
    """
    Assert that inference works but with a warning.
    """
    pyro.clear_param_store()
    inference = SVI(model,  guide, Adam({"lr": 1e-6}), elbo)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        inference.step()
        assert len(w), 'No warnings were raised'
        for warning in w:
            logger.info(warning)


@pytest.mark.parametrize("Elbo", [
    Trace_ELBO,
    TraceGraph_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    EnergyDistance_prior,
    EnergyDistance_noprior,
])
@pytest.mark.parametrize("strict_enumeration_warning", [True, False])
def test_nonempty_model_empty_guide_ok(Elbo, strict_enumeration_warning):

    def model():
        loc = torch.tensor([0.0, 0.0])
        scale = torch.tensor([1.0, 1.0])
        pyro.sample("x", dist.Normal(loc, scale).to_event(1), obs=loc)

    def guide():
        pass

    elbo = Elbo(strict_enumeration_warning=strict_enumeration_warning)
    if strict_enumeration_warning and Elbo in (TraceEnum_ELBO, TraceTMC_ELBO):
        assert_warning(model, guide, elbo)
    else:
        assert_ok(model, guide, elbo)


@pytest.mark.parametrize("Elbo", [
    Trace_ELBO,
    TraceGraph_ELBO,
    TraceEnum_ELBO,
    TraceTMC_ELBO,
    EnergyDistance_prior,
    EnergyDistance_noprior,
])
@pytest.mark.parametrize("strict_enumeration_warning", [True, False])
def test_nonempty_model_empty_guide_error(Elbo, strict_enumeration_warning):

    def model():
        pyro.sample("x", dist.Normal(0, 1))

    def guide():
        pass

    elbo = Elbo(strict_enumeration_warning=strict_enumeration_warning)
    assert_error(model, guide, elbo)


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
@pytest.mark.parametrize("strict_enumeration_warning", [True, False])
def test_empty_model_empty_guide_ok(Elbo, strict_enumeration_warning):

    def model():
        pass

    def guide():
        pass

    elbo = Elbo(strict_enumeration_warning=strict_enumeration_warning)
    if strict_enumeration_warning and Elbo in (TraceEnum_ELBO, TraceTMC_ELBO):
        assert_warning(model, guide, elbo)
    else:
        assert_ok(model, guide, elbo)


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_variable_clash_in_model_error(Elbo):

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p))
        pyro.sample("x", dist.Bernoulli(p))  # Should error here.

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))

    assert_error(model, guide, Elbo(), match='Multiple sample sites named')


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_model_guide_dim_mismatch_error(Elbo):

    def model():
        loc = torch.zeros(2)
        scale = torch.ones(2)
        pyro.sample("x", dist.Normal(loc, scale).to_event(1))

    def guide():
        loc = pyro.param("loc", torch.zeros(2, 1, requires_grad=True))
        scale = pyro.param("scale", torch.ones(2, 1, requires_grad=True))
        pyro.sample("x", dist.Normal(loc, scale).to_event(2))

    assert_error(model, guide, Elbo(strict_enumeration_warning=False),
                 match='invalid log_prob shape|Model and guide event_dims disagree')


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_model_guide_shape_mismatch_error(Elbo):

    def model():
        loc = torch.zeros(1, 2)
        scale = torch.ones(1, 2)
        pyro.sample("x", dist.Normal(loc, scale).to_event(2))

    def guide():
        loc = pyro.param("loc", torch.zeros(2, 1, requires_grad=True))
        scale = pyro.param("scale", torch.ones(2, 1, requires_grad=True))
        pyro.sample("x", dist.Normal(loc, scale).to_event(2))

    assert_error(model, guide, Elbo(strict_enumeration_warning=False),
                 match='Model and guide shapes disagree')


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_variable_clash_in_guide_error(Elbo):

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))
        pyro.sample("x", dist.Bernoulli(p))  # Should error here.

    assert_error(model, guide, Elbo(), match='Multiple sample sites named')


@pytest.mark.parametrize("has_rsample", [False, True])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_set_has_rsample_ok(has_rsample, Elbo):

    # This model has sparse gradients, so users may want to disable
    # reparametrized sampling to reduce variance of gradient estimates.
    # However both versions should be correct, i.e. with or without has_rsample.
    def model():
        z = pyro.sample("z", dist.Normal(0, 1))
        loc = (z * 100).clamp(min=0, max=1)  # sparse gradients
        pyro.sample("x", dist.Normal(loc, 1), obs=torch.tensor(0.))

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        pyro.sample("z", dist.Normal(loc, 1).has_rsample_(has_rsample))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo(strict_enumeration_warning=False))


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_not_has_rsample_ok(Elbo):

    def model():
        z = pyro.sample("z", dist.Normal(0, 1))
        p = z.round().clamp(min=0.2, max=0.8)  # discontinuous
        pyro.sample("x", dist.Bernoulli(p), obs=torch.tensor(0.))

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        pyro.sample("z", dist.Normal(loc, 1).has_rsample_(False))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo(strict_enumeration_warning=False))


@pytest.mark.parametrize("subsample_size", [None, 2], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_ok(subsample_size, Elbo):

    def model():
        p = torch.tensor(0.5)
        for i in pyro.plate("plate", 4, subsample_size):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        for i in pyro.plate("plate", 4, subsample_size):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_variable_clash_error(Elbo):

    def model():
        p = torch.tensor(0.5)
        for i in pyro.plate("plate", 2):
            # Each loop iteration should give the sample site a different name.
            pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        for i in pyro.plate("plate", 2):
            # Each loop iteration should give the sample site a different name.
            pyro.sample("x", dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_error(model, guide, Elbo(), match='Multiple sample sites named')


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_ok(subsample_size, Elbo):

    def model():
        p = torch.tensor(0.5)
        with pyro.plate("plate", 10, subsample_size) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.plate("plate", 10, subsample_size) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind)]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_subsample_param_ok(subsample_size, Elbo):

    def model():
        p = torch.tensor(0.5)
        with pyro.plate("plate", 10, subsample_size):
            pyro.sample("x", dist.Bernoulli(p))

    def guide():
        with pyro.plate("plate", 10, subsample_size) as ind:
            p0 = pyro.param("p0", torch.tensor(0.), event_dim=0)
            assert p0.shape == ()
            p = pyro.param("p", 0.5 * torch.ones(10), event_dim=0)
            assert len(p) == len(ind)
            pyro.sample("x", dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_subsample_primitive_ok(subsample_size, Elbo):

    def model():
        p = torch.tensor(0.5)
        with pyro.plate("plate", 10, subsample_size):
            pyro.sample("x", dist.Bernoulli(p))

    def guide():
        with pyro.plate("plate", 10, subsample_size) as ind:
            p0 = torch.tensor(0.)
            p0 = pyro.subsample(p0, event_dim=0)
            assert p0.shape == ()
            p = 0.5 * torch.ones(10)
            p = pyro.subsample(p, event_dim=0)
            assert len(p) == len(ind)
            pyro.sample("x", dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
@pytest.mark.parametrize("shape,ok", [
    ((), True),
    ((1,), True),
    ((10,), True),
    ((3, 1), True),
    ((3, 10), True),
    ((5), False),
    ((3, 5), False),
])
def test_plate_param_size_mismatch_error(subsample_size, Elbo, shape, ok):

    def model():
        p = torch.tensor(0.5)
        with pyro.plate("plate", 10, subsample_size):
            pyro.sample("x", dist.Bernoulli(p))

    def guide():
        with pyro.plate("plate", 10, subsample_size):
            pyro.param("p0", torch.ones(shape), event_dim=0)
            p = pyro.param("p", torch.ones(10), event_dim=0)
            pyro.sample("x", dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    if ok:
        assert_ok(model, guide, Elbo())
    else:
        assert_error(model, guide, Elbo(), match="invalid shape of pyro.param")


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_no_size_ok(Elbo):

    def model():
        p = torch.tensor(0.5)
        with pyro.plate("plate"):
            pyro.sample("x", dist.Bernoulli(p).expand_by([10]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.plate("plate"):
            pyro.sample("x", dist.Bernoulli(p).expand_by([10]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, default="parallel", num_samples=2)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("max_plate_nesting", [0, float('inf')])
@pytest.mark.parametrize("subsample_size", [None, 2], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_iplate_ok(subsample_size, Elbo, max_plate_nesting):

    def model():
        p = torch.tensor(0.5)
        outer_iplate = pyro.plate("plate_0", 3, subsample_size)
        inner_iplate = pyro.plate("plate_1", 3, subsample_size)
        for i in outer_iplate:
            for j in inner_iplate:
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        outer_iplate = pyro.plate("plate_0", 3, subsample_size)
        inner_iplate = pyro.plate("plate_1", 3, subsample_size)
        for i in outer_iplate:
            for j in inner_iplate:
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide, "parallel")
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo(max_plate_nesting=max_plate_nesting))


@pytest.mark.parametrize("max_plate_nesting", [0, float('inf')])
@pytest.mark.parametrize("subsample_size", [None, 2], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_iplate_swap_ok(subsample_size, Elbo, max_plate_nesting):

    def model():
        p = torch.tensor(0.5)
        outer_iplate = pyro.plate("plate_0", 3, subsample_size)
        inner_iplate = pyro.plate("plate_1", 3, subsample_size)
        for i in outer_iplate:
            for j in inner_iplate:
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        outer_iplate = pyro.plate("plate_0", 3, subsample_size)
        inner_iplate = pyro.plate("plate_1", 3, subsample_size)
        for j in inner_iplate:
            for i in outer_iplate:
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide, "parallel")
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, default="parallel", num_samples=2)

    assert_ok(model, guide, Elbo(max_plate_nesting=max_plate_nesting))


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_in_model_not_guide_ok(subsample_size, Elbo):

    def model():
        p = torch.tensor(0.5)
        for i in pyro.plate("plate", 10, subsample_size):
            pass
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
@pytest.mark.parametrize("is_validate", [True, False])
def test_iplate_in_guide_not_model_error(subsample_size, Elbo, is_validate):

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        for i in pyro.plate("plate", 10, subsample_size):
            pass
        pyro.sample("x", dist.Bernoulli(p))

    with pyro.validation_enabled(is_validate):
        if is_validate:
            assert_error(model, guide, Elbo(),
                         match='Found plate statements in guide but not model')
        else:
            assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_plate_broadcast_error(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate("plate", 10, 5):
            pyro.sample("x", dist.Bernoulli(p).expand_by([2]))

    assert_error(model, model, Elbo(), match='Shape mismatch inside plate')


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_iplate_ok(Elbo):

    def model():
        p = torch.tensor(0.5)
        with pyro.plate("plate", 3, 2) as ind:
            for i in pyro.plate("iplate", 3, 2):
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.plate("plate", 3, 2) as ind:
            for i in pyro.plate("iplate", 3, 2):
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_iplate_plate_ok(Elbo):

    def model():
        p = torch.tensor(0.5)
        inner_plate = pyro.plate("plate", 3, 2)
        for i in pyro.plate("iplate", 3, 2):
            with inner_plate as ind:
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        inner_plate = pyro.plate("plate", 3, 2)
        for i in pyro.plate("iplate", 3, 2):
            with inner_plate as ind:
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
@pytest.mark.parametrize("sizes", [(3,), (3, 4), (3, 4, 5)])
def test_plate_stack_ok(Elbo, sizes):

    def model():
        p = torch.tensor(0.5)
        with pyro.plate_stack("plate_stack", sizes):
            pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.plate_stack("plate_stack", sizes):
            pyro.sample("x", dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
@pytest.mark.parametrize("sizes", [(3,), (3, 4), (3, 4, 5)])
def test_plate_stack_and_plate_ok(Elbo, sizes):

    def model():
        p = torch.tensor(0.5)
        with pyro.plate_stack("plate_stack", sizes):
            with pyro.plate("plate", 7):
                pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.plate_stack("plate_stack", sizes):
            with pyro.plate("plate", 7):
                pyro.sample("x", dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(guide, num_samples=2)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("sizes", [(3,), (3, 4), (3, 4, 5)])
def test_plate_stack_sizes(sizes):

    def model():
        p = 0.5 * torch.ones(3)
        with pyro.plate_stack("plate_stack", sizes):
            x = pyro.sample("x", dist.Bernoulli(p).to_event(1))
            assert x.shape == sizes + (3,)

    model()


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_nested_plate_plate_ok(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate("plate_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer)]))
            with pyro.plate("plate_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner), len(ind_outer)]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(model)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(model, num_samples=2)
    else:
        guide = model

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_plate_reuse_ok(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        plate_outer = pyro.plate("plate_outer", 10, 5, dim=-1)
        plate_inner = pyro.plate("plate_inner", 11, 6, dim=-2)
        with plate_outer as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer)]))
        with plate_inner as ind_inner:
            pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner), 1]))
        with plate_outer as ind_outer, plate_inner as ind_inner:
            pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_inner), len(ind_outer)]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(model)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(model, num_samples=2)
    else:
        guide = model

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO, TraceTMC_ELBO])
def test_nested_plate_plate_dim_error_1(Elbo):

    def model():
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.plate("plate_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer)]))  # error here
            with pyro.plate("plate_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner)]))
                pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_outer), len(ind_inner)]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(model)
    elif Elbo is TraceTMC_ELBO:
        guide = config_enumerate(model, num_samples=2)
    else:
        guide = model

    assert_error(model, guide, Elbo(), match='invalid log_prob shape')


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_plate_plate_dim_error_2(Elbo):

    def model():
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.plate("plate_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer), 1]))
            with pyro.plate("plate_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_outer)]))  # error here
                pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_outer), len(ind_inner)]))

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo(), match='Shape mismatch inside plate')


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_plate_plate_dim_error_3(Elbo):

    def model():
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.plate("plate_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer), 1]))
            with pyro.plate("plate_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner)]))
                pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_inner), 1]))  # error here

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo(), match='invalid log_prob shape|shape mismatch')


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_plate_plate_dim_error_4(Elbo):

    def model():
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.plate("plate_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer), 1]))
            with pyro.plate("plate_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner)]))
                pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_outer), len(ind_outer)]))  # error here

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo(), match='hape mismatch inside plate')


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_plate_plate_subsample_param_ok(Elbo):

    def model():
        with pyro.plate("plate_outer", 10, 5):
            pyro.sample("x", dist.Bernoulli(0.2))
            with pyro.plate("plate_inner", 11, 6):
                pyro.sample("y", dist.Bernoulli(0.2))

    def guide():
        p0 = pyro.param("p0", 0.5 * torch.ones(4, 5), event_dim=2)
        assert p0.shape == (4, 5)
        with pyro.plate("plate_outer", 10, 5):
            p1 = pyro.param("p1", 0.5 * torch.ones(10, 3), event_dim=1)
            assert p1.shape == (5, 3)
            px = pyro.param("px", 0.5 * torch.ones(10), event_dim=0)
            assert px.shape == (5,)
            pyro.sample("x", dist.Bernoulli(px))
            with pyro.plate("plate_inner", 11, 6):
                py = pyro.param("py", 0.5 * torch.ones(11, 10), event_dim=0)
                assert py.shape == (6, 5)
                pyro.sample("y", dist.Bernoulli(py))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)
    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nonnested_plate_plate_ok(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate("plate_0", 10, 5) as ind1:
            pyro.sample("x0", dist.Bernoulli(p).expand_by([len(ind1)]))
        with pyro.plate("plate_1", 11, 6) as ind2:
            pyro.sample("x1", dist.Bernoulli(p).expand_by([len(ind2)]))

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_ok(model, guide, Elbo())


def test_three_indep_plate_at_different_depths_ok():
    r"""
      /\
     /\ ia
    ia ia
    """
    def model():
        p = torch.tensor(0.5)
        inner_plate = pyro.plate("plate2", 10, 5)
        for i in pyro.plate("plate0", 2):
            pyro.sample("x_%d" % i, dist.Bernoulli(p))
            if i == 0:
                for j in pyro.plate("plate1", 2):
                    with inner_plate as ind:
                        pyro.sample("y_%d" % j, dist.Bernoulli(p).expand_by([len(ind)]))
            elif i == 1:
                with inner_plate as ind:
                    pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        inner_plate = pyro.plate("plate2", 10, 5)
        for i in pyro.plate("plate0", 2):
            pyro.sample("x_%d" % i, dist.Bernoulli(p))
            if i == 0:
                for j in pyro.plate("plate1", 2):
                    with inner_plate as ind:
                        pyro.sample("y_%d" % j, dist.Bernoulli(p).expand_by([len(ind)]))
            elif i == 1:
                with inner_plate as ind:
                    pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind)]))

    assert_ok(model, guide, TraceGraph_ELBO())


def test_plate_wrong_size_error():

    def model():
        p = torch.tensor(0.5)
        with pyro.plate("plate", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([1 + len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.plate("plate", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([1 + len(ind)]))

    assert_error(model, guide, TraceGraph_ELBO(), match='Shape mismatch inside plate')


@pytest.mark.parametrize("enumerate_", [None, "sequential", "parallel"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_enum_discrete_misuse_warning(Elbo, enumerate_):

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p), infer={"enumerate": enumerate_})

    if (enumerate_ is None) == (Elbo is TraceEnum_ELBO):
        assert_warning(model, guide, Elbo(max_plate_nesting=0))
    else:
        assert_ok(model, guide, Elbo(max_plate_nesting=0))


def test_enum_discrete_single_ok():

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))

    assert_ok(model, config_enumerate(guide), TraceEnum_ELBO())


@pytest.mark.parametrize("strict_enumeration_warning", [False, True])
def test_enum_discrete_missing_config_warning(strict_enumeration_warning):

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))

    elbo = TraceEnum_ELBO(strict_enumeration_warning=strict_enumeration_warning)
    if strict_enumeration_warning:
        assert_warning(model, guide, elbo)
    else:
        assert_ok(model, guide, elbo)


def test_enum_discrete_single_single_ok():

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p))
        pyro.sample("y", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))
        pyro.sample("y", dist.Bernoulli(p))

    assert_ok(model, config_enumerate(guide), TraceEnum_ELBO())


def test_enum_discrete_iplate_single_ok():

    def model():
        p = torch.tensor(0.5)
        for i in pyro.plate("plate", 10, 5):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        for i in pyro.plate("plate", 10, 5):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    assert_ok(model, config_enumerate(guide), TraceEnum_ELBO())


def test_plate_enum_discrete_batch_ok():

    def model():
        p = torch.tensor(0.5)
        with pyro.plate("plate", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.plate("plate", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind)]))

    assert_ok(model, config_enumerate(guide), TraceEnum_ELBO())


@pytest.mark.parametrize("strict_enumeration_warning", [False, True])
def test_plate_enum_discrete_no_discrete_vars_warning(strict_enumeration_warning):

    def model():
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        with pyro.plate("plate", 10, 5) as ind:
            pyro.sample("x", dist.Normal(loc, scale).expand_by([len(ind)]))

    @config_enumerate(default="sequential")
    def guide():
        loc = pyro.param("loc", torch.tensor(1.0, requires_grad=True))
        scale = pyro.param("scale", torch.tensor(2.0, requires_grad=True))
        with pyro.plate("plate", 10, 5) as ind:
            pyro.sample("x", dist.Normal(loc, scale).expand_by([len(ind)]))

    elbo = TraceEnum_ELBO(strict_enumeration_warning=strict_enumeration_warning)
    if strict_enumeration_warning:
        assert_warning(model, guide, elbo)
    else:
        assert_ok(model, guide, elbo)


def test_no_plate_enum_discrete_batch_error():

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p).expand_by([5]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p).expand_by([5]))

    assert_error(model, config_enumerate(guide), TraceEnum_ELBO(),
                 match='invalid log_prob shape')


@pytest.mark.parametrize('max_plate_nesting', [0, 1, 2, float('inf')])
def test_enum_discrete_parallel_ok(max_plate_nesting):
    guessed_nesting = 0 if max_plate_nesting == float('inf') else max_plate_nesting
    plate_shape = torch.Size([1] * guessed_nesting)

    def model():
        p = torch.tensor(0.5)
        x = pyro.sample("x", dist.Bernoulli(p))
        if max_plate_nesting != float('inf'):
            assert x.shape == torch.Size([2]) + plate_shape

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        x = pyro.sample("x", dist.Bernoulli(p))
        if max_plate_nesting != float('inf'):
            assert x.shape == torch.Size([2]) + plate_shape

    assert_ok(model, config_enumerate(guide, "parallel"),
              TraceEnum_ELBO(max_plate_nesting=max_plate_nesting))


@pytest.mark.parametrize('max_plate_nesting', [0, 1, 2, float('inf')])
def test_enum_discrete_parallel_nested_ok(max_plate_nesting):
    guessed_nesting = 0 if max_plate_nesting == float('inf') else max_plate_nesting
    plate_shape = torch.Size([1] * guessed_nesting)

    def model():
        p2 = torch.ones(2) / 2
        p3 = torch.ones(3) / 3
        x2 = pyro.sample("x2", dist.OneHotCategorical(p2))
        x3 = pyro.sample("x3", dist.OneHotCategorical(p3))
        if max_plate_nesting != float('inf'):
            assert x2.shape == torch.Size([2]) + plate_shape + p2.shape
            assert x3.shape == torch.Size([3, 1]) + plate_shape + p3.shape

    assert_ok(model, config_enumerate(model, "parallel"),
              TraceEnum_ELBO(max_plate_nesting=max_plate_nesting))


@pytest.mark.parametrize('enumerate_,expand,num_samples', [
    (None, False, None),
    ("sequential", False, None),
    ("sequential", True, None),
    ("parallel", False, None),
    ("parallel", True, None),
    ("parallel", True, 3),
])
def test_enumerate_parallel_plate_ok(enumerate_, expand, num_samples):

    def model():
        p2 = torch.ones(2) / 2
        p34 = torch.ones(3, 4) / 4
        p536 = torch.ones(5, 3, 6) / 6

        x2 = pyro.sample("x2", dist.Categorical(p2))
        with pyro.plate("outer", 3):
            x34 = pyro.sample("x34", dist.Categorical(p34))
            with pyro.plate("inner", 5):
                x536 = pyro.sample("x536", dist.Categorical(p536))

        # check shapes
        if enumerate_ == "parallel":
            if num_samples:
                n = num_samples
                # Meaning of dimensions:    [ enum dims | plate dims ]
                assert x2.shape == torch.Size([        n, 1, 1])  # noqa: E201
                assert x34.shape == torch.Size([    n, 1, 1, 3])  # noqa: E201
                assert x536.shape == torch.Size([n, 1, 1, 5, 3])  # noqa: E201
            elif expand:
                # Meaning of dimensions:    [ enum dims | plate dims ]
                assert x2.shape == torch.Size([        2, 1, 1])  # noqa: E201
                assert x34.shape == torch.Size([    4, 1, 1, 3])  # noqa: E201
                assert x536.shape == torch.Size([6, 1, 1, 5, 3])  # noqa: E201
            else:
                # Meaning of dimensions:    [ enum dims | plate placeholders ]
                assert x2.shape == torch.Size([        2, 1, 1])  # noqa: E201
                assert x34.shape == torch.Size([    4, 1, 1, 1])  # noqa: E201
                assert x536.shape == torch.Size([6, 1, 1, 1, 1])  # noqa: E201
        elif enumerate_ == "sequential":
            if expand:
                # All dimensions are plate dimensions.
                assert x2.shape == torch.Size([])
                assert x34.shape == torch.Size([3])
                assert x536.shape == torch.Size([5, 3])
            else:
                # All dimensions are plate placeholders.
                assert x2.shape == torch.Size([])
                assert x34.shape == torch.Size([1])
                assert x536.shape == torch.Size([1, 1])
        else:
            # All dimensions are plate dimensions.
            assert x2.shape == torch.Size([])
            assert x34.shape == torch.Size([3])
            assert x536.shape == torch.Size([5, 3])

    elbo = TraceEnum_ELBO(max_plate_nesting=2, strict_enumeration_warning=enumerate_)
    guide = config_enumerate(model, enumerate_, expand, num_samples)
    assert_ok(model, guide, elbo)


@pytest.mark.parametrize('max_plate_nesting', [1, float('inf')])
@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
@pytest.mark.parametrize('is_validate', [True, False])
def test_enum_discrete_plate_dependency_warning(enumerate_, is_validate, max_plate_nesting):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        with pyro.plate("plate", 10, 5):
            x = pyro.sample("x", dist.Bernoulli(0.5).expand_by([5]),
                            infer={'enumerate': enumerate_})
        pyro.sample("y", dist.Bernoulli(x.mean()))  # user should move this line up

    with pyro.validation_enabled(is_validate):
        elbo = TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)
        if enumerate_ and is_validate:
            assert_warning(model, model, elbo)
        else:
            assert_ok(model, model, elbo)


@pytest.mark.parametrize('max_plate_nesting', [1, float('inf')])
@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
def test_enum_discrete_iplate_plate_dependency_ok(enumerate_, max_plate_nesting):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        inner_plate = pyro.plate("plate", 10, 5)
        for i in pyro.plate("iplate", 3):
            pyro.sample("y_{}".format(i), dist.Bernoulli(0.5))
            with inner_plate:
                pyro.sample("x_{}".format(i), dist.Bernoulli(0.5).expand_by([5]),
                            infer={'enumerate': enumerate_})

    assert_ok(model, model, TraceEnum_ELBO(max_plate_nesting=max_plate_nesting))


@pytest.mark.parametrize('max_plate_nesting', [1, float('inf')])
@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
@pytest.mark.parametrize('is_validate', [True, False])
def test_enum_discrete_iplates_plate_dependency_warning(enumerate_, is_validate, max_plate_nesting):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        inner_plate = pyro.plate("plate", 10, 5)

        for i in pyro.plate("iplate1", 2):
            with inner_plate:
                pyro.sample("x_{}".format(i), dist.Bernoulli(0.5).expand_by([5]),
                            infer={'enumerate': enumerate_})

        for i in pyro.plate("iplate2", 2):
            pyro.sample("y_{}".format(i), dist.Bernoulli(0.5))

    with pyro.validation_enabled(is_validate):
        elbo = TraceEnum_ELBO(max_plate_nesting=max_plate_nesting)
        if enumerate_ and is_validate:
            assert_warning(model, model, elbo)
        else:
            assert_ok(model, model, elbo)


@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
def test_enum_discrete_plates_dependency_ok(enumerate_):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        x_plate = pyro.plate("x_plate", 10, 5, dim=-1)
        y_plate = pyro.plate("y_plate", 11, 6, dim=-2)
        pyro.sample("a", dist.Bernoulli(0.5))
        with x_plate:
            pyro.sample("b", dist.Bernoulli(0.5).expand_by([5]))
        with y_plate:
            # Note that it is difficult to check that c does not depend on b.
            pyro.sample("c", dist.Bernoulli(0.5).expand_by([6, 1]))
        with x_plate, y_plate:
            pyro.sample("d", dist.Bernoulli(0.5).expand_by([6, 5]))

    assert_ok(model, model, TraceEnum_ELBO(max_plate_nesting=2))


@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
def test_enum_discrete_non_enumerated_plate_ok(enumerate_):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})

        with pyro.plate("non_enum", 2):
            a = pyro.sample("a", dist.Bernoulli(0.5).expand_by([2]),
                            infer={'enumerate': None})

        p = (1.0 + a.sum(-1)) / (2.0 + a.size(0))  # introduce dependency of b on a

        with pyro.plate("enum_1", 3):
            pyro.sample("b", dist.Bernoulli(p).expand_by([3]),
                        infer={'enumerate': enumerate_})

    with pyro.validation_enabled():
        assert_ok(model, model, TraceEnum_ELBO(max_plate_nesting=1))


def test_plate_shape_broadcasting():
    data = torch.ones(1000, 2)

    def model():
        with pyro.plate("num_particles", 10, dim=-3):
            with pyro.plate("components", 2, dim=-1):
                p = pyro.sample("p", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
                assert p.shape == torch.Size((10, 1, 2))
            with pyro.plate("data", data.shape[0], dim=-2):
                pyro.sample("obs", dist.Bernoulli(p), obs=data)

    def guide():
        with pyro.plate("num_particles", 10, dim=-3):
            with pyro.plate("components", 2, dim=-1):
                pyro.sample("p", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))

    assert_ok(model, guide, Trace_ELBO())


@pytest.mark.parametrize('enumerate_,expand,num_samples', [
    (None, True, None),
    ("sequential", True, None),
    ("sequential", False, None),
    ("parallel", True, None),
    ("parallel", False, None),
    ("parallel", True, 3),
])
def test_enum_discrete_plate_shape_broadcasting_ok(enumerate_, expand, num_samples):

    def model():
        x_plate = pyro.plate("x_plate", 10, 5, dim=-1)
        y_plate = pyro.plate("y_plate", 11, 6, dim=-2)
        with pyro.plate("num_particles", 50, dim=-3):
            with x_plate:
                b = pyro.sample("b", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
            with y_plate:
                c = pyro.sample("c", dist.Bernoulli(0.5))
            with x_plate, y_plate:
                d = pyro.sample("d", dist.Bernoulli(b))

        # check shapes
        if enumerate_ == "parallel":
            if num_samples and expand:
                assert b.shape == (num_samples, 50, 1, 5)
                assert c.shape == (num_samples, 1, 50, 6, 1)
                assert d.shape == (num_samples, 1, num_samples, 50, 6, 5)
            elif num_samples and not expand:
                assert b.shape == (num_samples, 50, 1, 5)
                assert c.shape == (num_samples, 1, 50, 6, 1)
                assert d.shape == (num_samples, 1, 1, 50, 6, 5)
            elif expand:
                assert b.shape == (50, 1, 5)
                assert c.shape == (2, 50, 6, 1)
                assert d.shape == (2, 1, 50, 6, 5)
            else:
                assert b.shape == (50, 1, 5)
                assert c.shape == (2, 1, 1, 1)
                assert d.shape == (2, 1, 1, 1, 1)
        elif enumerate_ == "sequential":
            if expand:
                assert b.shape == (50, 1, 5)
                assert c.shape == (50, 6, 1)
                assert d.shape == (50, 6, 5)
            else:
                assert b.shape == (50, 1, 5)
                assert c.shape == (1, 1, 1)
                assert d.shape == (1, 1, 1)
        else:
            assert b.shape == (50, 1, 5)
            assert c.shape == (50, 6, 1)
            assert d.shape == (50, 6, 5)

    guide = config_enumerate(model, default=enumerate_, expand=expand, num_samples=num_samples)
    elbo = TraceEnum_ELBO(max_plate_nesting=3,
                          strict_enumeration_warning=(enumerate_ == "parallel"))
    assert_ok(model, guide, elbo)


@pytest.mark.parametrize("Elbo,expand", [
    (Trace_ELBO, False),
    (TraceGraph_ELBO, False),
    (TraceEnum_ELBO, False),
    (TraceEnum_ELBO, True),
])
def test_dim_allocation_ok(Elbo, expand):
    enumerate_ = (Elbo is TraceEnum_ELBO)

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate("plate_outer", 10, 5, dim=-3):
            x = pyro.sample("x", dist.Bernoulli(p))
            with pyro.plate("plate_inner_1", 11, 6):
                y = pyro.sample("y", dist.Bernoulli(p))
                # allocated dim is rightmost available, i.e. -1
                with pyro.plate("plate_inner_2", 12, 7):
                    z = pyro.sample("z", dist.Bernoulli(p))
                    # allocated dim is next rightmost available, i.e. -2
                    # since dim -3 is already allocated, use dim=-4
                    with pyro.plate("plate_inner_3", 13, 8):
                        q = pyro.sample("q", dist.Bernoulli(p))

        # check shapes
        if enumerate_ and not expand:
            assert x.shape == (1, 1, 1)
            assert y.shape == (1, 1, 1)
            assert z.shape == (1, 1, 1)
            assert q.shape == (1, 1, 1, 1)
        else:
            assert x.shape == (5, 1, 1)
            assert y.shape == (5, 1, 6)
            assert z.shape == (5, 7, 6)
            assert q.shape == (8, 5, 7, 6)

    guide = config_enumerate(model, "sequential", expand=expand) if enumerate_ else model
    assert_ok(model, guide, Elbo(max_plate_nesting=4))


@pytest.mark.parametrize("Elbo,expand", [
    (Trace_ELBO, False),
    (TraceGraph_ELBO, False),
    (TraceEnum_ELBO, False),
    (TraceEnum_ELBO, True),
])
def test_dim_allocation_error(Elbo, expand):
    enumerate_ = (Elbo is TraceEnum_ELBO)

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate("plate_outer", 10, 5, dim=-2):
            x = pyro.sample("x", dist.Bernoulli(p))
            # allocated dim is rightmost available, i.e. -1
            with pyro.plate("plate_inner_1", 11, 6):
                y = pyro.sample("y", dist.Bernoulli(p))
                # throws an error as dim=-1 is already occupied
                with pyro.plate("plate_inner_2", 12, 7, dim=-1):
                    pyro.sample("z", dist.Bernoulli(p))

        # check shapes
        if enumerate_ and not expand:
            assert x.shape == (1, 1)
            assert y.shape == (1, 1)
        else:
            assert x.shape == (5, 1)
            assert y.shape == (5, 6)

    guide = config_enumerate(model, expand=expand) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo(), match='collide at dim=')


def test_enum_in_model_ok():
    infer = {'enumerate': 'parallel'}

    def model():
        p = pyro.param('p', torch.tensor(0.25))
        a = pyro.sample('a', dist.Bernoulli(p))
        b = pyro.sample('b', dist.Bernoulli(p + a / 2))
        c = pyro.sample('c', dist.Bernoulli(p + b / 2), infer=infer)
        d = pyro.sample('d', dist.Bernoulli(p + c / 2))
        e = pyro.sample('e', dist.Bernoulli(p + d / 2))
        f = pyro.sample('f', dist.Bernoulli(p + e / 2), infer=infer)
        g = pyro.sample('g', dist.Bernoulli(p + f / 2), obs=torch.tensor(0.))

        # check shapes
        assert a.shape == ()
        assert b.shape == (2,)
        assert c.shape == (2, 1, 1)
        assert d.shape == (2,)
        assert e.shape == (2, 1)
        assert f.shape == (2, 1, 1, 1)
        assert g.shape == ()

    def guide():
        p = pyro.param('p', torch.tensor(0.25))
        a = pyro.sample('a', dist.Bernoulli(p))
        b = pyro.sample('b', dist.Bernoulli(p + a / 2), infer=infer)
        d = pyro.sample('d', dist.Bernoulli(p + b / 2))
        e = pyro.sample('e', dist.Bernoulli(p + d / 2), infer=infer)

        # check shapes
        assert a.shape == ()
        assert b.shape == (2,)
        assert d.shape == (2,)
        assert e.shape == (2, 1)

    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))


def test_enum_in_model_plate_ok():
    infer = {'enumerate': 'parallel'}

    def model():
        p = pyro.param('p', torch.tensor(0.25))
        a = pyro.sample('a', dist.Bernoulli(p))
        b = pyro.sample('b', dist.Bernoulli(p + a / 2))
        with pyro.plate('data', 3):
            c = pyro.sample('c', dist.Bernoulli(p + b / 2), infer=infer)
            d = pyro.sample('d', dist.Bernoulli(p + c / 2))
            e = pyro.sample('e', dist.Bernoulli(p + d / 2))
            f = pyro.sample('f', dist.Bernoulli(p + e / 2), infer=infer)
            g = pyro.sample('g', dist.Bernoulli(p + f / 2), obs=torch.zeros(3))

        # check shapes
        assert a.shape == ()
        assert b.shape == (2, 1)
        assert c.shape == (2, 1, 1, 1)
        assert d.shape == (2, 3)
        assert e.shape == (2, 1, 1)
        assert f.shape == (2, 1, 1, 1, 1)
        assert g.shape == (3,)

    def guide():
        p = pyro.param('p', torch.tensor(0.25))
        a = pyro.sample('a', dist.Bernoulli(p))
        b = pyro.sample('b', dist.Bernoulli(p + a / 2), infer=infer)
        with pyro.plate('data', 3):
            d = pyro.sample('d', dist.Bernoulli(p + b / 2))
            e = pyro.sample('e', dist.Bernoulli(p + d / 2), infer=infer)

        # check shapes
        assert a.shape == ()
        assert b.shape == (2, 1)
        assert d.shape == (2, 3)
        assert e.shape == (2, 1, 1)

    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=1))


def test_enum_sequential_in_model_error():

    def model():
        p = pyro.param('p', torch.tensor(0.25))
        pyro.sample('a', dist.Bernoulli(p), infer={'enumerate': 'sequential'})

    def guide():
        pass

    assert_error(model, guide, TraceEnum_ELBO(max_plate_nesting=0),
                 match='Found vars in model but not guide')


def test_enum_in_model_plate_reuse_ok():

    @config_enumerate
    def model():
        p = pyro.param("p", torch.tensor([0.2, 0.8]))
        a = pyro.sample("a", dist.Bernoulli(0.3)).long()
        with pyro.plate("b_axis", 2):
            pyro.sample("b", dist.Bernoulli(p[a]), obs=torch.tensor([0., 1.]))
        c = pyro.sample("c", dist.Bernoulli(0.3)).long()
        with pyro.plate("c_axis", 2):
            pyro.sample("d", dist.Bernoulli(p[c]), obs=torch.tensor([0., 0.]))

    def guide():
        pass

    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=1))


def test_enum_in_model_multi_scale_error():

    @config_enumerate
    def model():
        p = pyro.param("p", torch.tensor([0.2, 0.8]))
        x = pyro.sample("x", dist.Bernoulli(0.3)).long()
        with poutine.scale(scale=2.):
            pyro.sample("y", dist.Bernoulli(p[x]), obs=torch.tensor(0.))

    def guide():
        pass

    assert_error(model, guide, TraceEnum_ELBO(max_plate_nesting=0),
                 match='Expected all enumerated sample sites to share a common poutine.scale')


@pytest.mark.parametrize('use_vindex', [False, True])
def test_enum_in_model_diamond_error(use_vindex):
    data = torch.tensor([[0, 1], [0, 0]])

    @config_enumerate
    def model():
        pyro.param("probs_a", torch.tensor([0.45, 0.55]))
        pyro.param("probs_b", torch.tensor([[0.6, 0.4], [0.4, 0.6]]))
        pyro.param("probs_c", torch.tensor([[0.75, 0.25], [0.55, 0.45]]))
        pyro.param("probs_d", torch.tensor([[[0.4, 0.6], [0.3, 0.7]],
                                            [[0.3, 0.7], [0.2, 0.8]]]))
        probs_a = pyro.param("probs_a")
        probs_b = pyro.param("probs_b")
        probs_c = pyro.param("probs_c")
        probs_d = pyro.param("probs_d")
        b_axis = pyro.plate("b_axis", 2, dim=-1)
        c_axis = pyro.plate("c_axis", 2, dim=-2)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        with b_axis, c_axis:
            if use_vindex:
                probs = Vindex(probs_d)[b, c]
            else:
                d_ind = torch.arange(2, dtype=torch.long)
                probs = probs_d[b.unsqueeze(-1), c.unsqueeze(-1), d_ind]
            pyro.sample("d", dist.Categorical(probs), obs=data)

    def guide():
        pass

    assert_error(model, guide, TraceEnum_ELBO(max_plate_nesting=2),
                 match='Expected tree-structured plate nesting')


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_vectorized_num_particles(Elbo):
    data = torch.ones(1000, 2)

    def model():
        with pyro.plate("components", 2):
            p = pyro.sample("p", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
            assert p.shape == torch.Size((10, 1, 2))
            with pyro.plate("data", data.shape[0]):
                pyro.sample("obs", dist.Bernoulli(p), obs=data)

    def guide():
        with pyro.plate("components", 2):
            pyro.sample("p", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))

    pyro.clear_param_store()
    guide = config_enumerate(guide) if Elbo is TraceEnum_ELBO else guide
    assert_ok(model, guide, Elbo(num_particles=10,
                                 vectorize_particles=True,
                                 max_plate_nesting=2,
                                 strict_enumeration_warning=False))


@pytest.mark.parametrize('enumerate_,expand,num_samples', [
    (None, False, None),
    ("sequential", False, None),
    ("sequential", True, None),
    ("parallel", False, None),
    ("parallel", True, None),
    ("parallel", True, 3),
])
@pytest.mark.parametrize('num_particles', [1, 50])
def test_enum_discrete_vectorized_num_particles(enumerate_, expand, num_samples, num_particles):

    @config_enumerate(default=enumerate_, expand=expand, num_samples=num_samples)
    def model():
        x_plate = pyro.plate("x_plate", 10, 5, dim=-1)
        y_plate = pyro.plate("y_plate", 11, 6, dim=-2)
        with x_plate:
            b = pyro.sample("b", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
        with y_plate:
            c = pyro.sample("c", dist.Bernoulli(0.5))
        with x_plate, y_plate:
            d = pyro.sample("d", dist.Bernoulli(b))

        # check shapes
        if num_particles > 1:
            if enumerate_ == "parallel":
                if num_samples and expand:
                    assert b.shape == (num_samples, num_particles, 1, 5)
                    assert c.shape == (num_samples, 1, num_particles, 6, 1)
                    assert d.shape == (num_samples, 1, num_samples, num_particles, 6, 5)
                elif num_samples and not expand:
                    assert b.shape == (num_samples, num_particles, 1, 5)
                    assert c.shape == (num_samples, 1, num_particles, 6, 1)
                    assert d.shape == (num_samples, 1, 1, num_particles, 6, 5)
                elif expand:
                    assert b.shape == (num_particles, 1, 5)
                    assert c.shape == (2, num_particles, 6, 1)
                    assert d.shape == (2, 1, num_particles, 6, 5)
                else:
                    assert b.shape == (num_particles, 1, 5)
                    assert c.shape == (2, 1, 1, 1)
                    assert d.shape == (2, 1, 1, 1, 1)
            elif enumerate_ == "sequential":
                if expand:
                    assert b.shape == (num_particles, 1, 5)
                    assert c.shape == (num_particles, 6, 1)
                    assert d.shape == (num_particles, 6, 5)
                else:
                    assert b.shape == (num_particles, 1, 5)
                    assert c.shape == (1, 1, 1)
                    assert d.shape == (1, 1, 1)
            else:
                assert b.shape == (num_particles, 1, 5)
                assert c.shape == (num_particles, 6, 1)
                assert d.shape == (num_particles, 6, 5)
        else:
            if enumerate_ == "parallel":
                if num_samples and expand:
                    assert b.shape == (num_samples, 1, 5,)
                    assert c.shape == (num_samples, 1, 6, 1)
                    assert d.shape == (num_samples, 1, num_samples, 6, 5)
                elif num_samples and not expand:
                    assert b.shape == (num_samples, 1, 5,)
                    assert c.shape == (num_samples, 1, 6, 1)
                    assert d.shape == (num_samples, 1, 1, 6, 5)
                elif expand:
                    assert b.shape == (5,)
                    assert c.shape == (2, 6, 1)
                    assert d.shape == (2, 1, 6, 5)
                else:
                    assert b.shape == (5,)
                    assert c.shape == (2, 1, 1)
                    assert d.shape == (2, 1, 1, 1)
            elif enumerate_ == "sequential":
                if expand:
                    assert b.shape == (5,)
                    assert c.shape == (6, 1)
                    assert d.shape == (6, 5)
                else:
                    assert b.shape == (5,)
                    assert c.shape == (1, 1)
                    assert d.shape == (1, 1)
            else:
                assert b.shape == (5,)
                assert c.shape == (6, 1)
                assert d.shape == (6, 5)

    assert_ok(model, model, TraceEnum_ELBO(max_plate_nesting=2,
                                           num_particles=num_particles,
                                           vectorize_particles=True,
                                           strict_enumeration_warning=(enumerate_ == "parallel")))


def test_enum_recycling_chain():

    @config_enumerate
    def model():
        p = pyro.param("p", torch.tensor([[0.2, 0.8], [0.1, 0.9]]))

        x = 0
        for t in pyro.markov(range(100)):
            x = pyro.sample("x_{}".format(t), dist.Categorical(p[x]))
            assert x.dim() <= 2

    def guide():
        pass

    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))


@pytest.mark.parametrize('use_vindex', [False, True])
@pytest.mark.parametrize('markov', [False, True])
def test_enum_recycling_dbn(markov, use_vindex):
    #    x --> x --> x  enum "state"
    # y  |  y  |  y  |  enum "occlusion"
    #  \ |   \ |   \ |
    #    z     z     z  obs

    @config_enumerate
    def model():
        p = pyro.param("p", torch.ones(3, 3))
        q = pyro.param("q", torch.ones(2))
        r = pyro.param("r", torch.ones(3, 2, 4))

        x = 0
        times = pyro.markov(range(100)) if markov else range(11)
        for t in times:
            x = pyro.sample("x_{}".format(t), dist.Categorical(p[x]))
            y = pyro.sample("y_{}".format(t), dist.Categorical(q))
            if use_vindex:
                probs = Vindex(r)[x, y]
            else:
                z_ind = torch.arange(4, dtype=torch.long)
                probs = r[x.unsqueeze(-1), y.unsqueeze(-1), z_ind]
            pyro.sample("z_{}".format(t), dist.Categorical(probs),
                        obs=torch.tensor(0.))

    def guide():
        pass

    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))


def test_enum_recycling_nested():
    # (x)
    #   \
    #    y0---(y1)--(y2)
    #    |     |     |
    #   z00   z10   z20
    #    |     |     |
    #   z01   z11  (z21)
    #    |     |     |
    #   z02   z12   z22 <-- what can this depend on?
    #
    # markov dependencies
    # -------------------
    #   x:
    #  y0: x
    # z00: x y0
    # z01: x y0 z00
    # z02: x y0 z01
    #  y1: x y0
    # z10: x y0 y1
    # z11: x y0 y1 z10
    # z12: x y0 y1 z11
    #  y2: x y1
    # z20: x y1 y2
    # z21: x y1 y2 z20
    # z22: x y1 y2 z21

    @config_enumerate
    def model():
        p = pyro.param("p", torch.ones(3, 3))
        x = pyro.sample("x", dist.Categorical(p[0]))
        y = x
        for i in pyro.markov(range(10)):
            y = pyro.sample("y_{}".format(i), dist.Categorical(p[y]))
            z = y
            for j in pyro.markov(range(10)):
                z = pyro.sample("z_{}_{}".format(i, j), dist.Categorical(p[z]))

    def guide():
        pass

    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))


@pytest.mark.parametrize('use_vindex', [False, True])
def test_enum_recycling_grid(use_vindex):
    #  x---x---x---x    -----> i
    #  |   |   |   |   |
    #  x---x---x---x   |
    #  |   |   |   |   V
    #  x---x---x--(x)  j
    #  |   |   |   |
    #  x---x--(x)--x <-- what can this depend on?

    @config_enumerate
    def model():
        p = pyro.param("p_leaf", torch.ones(2, 2, 2))
        x = defaultdict(lambda: torch.tensor(0))
        y_axis = pyro.markov(range(4), keep=True)
        for i in pyro.markov(range(4)):
            for j in y_axis:
                if use_vindex:
                    probs = Vindex(p)[x[i - 1, j], x[i, j - 1]]
                else:
                    ind = torch.arange(2, dtype=torch.long)
                    probs = p[x[i - 1, j].unsqueeze(-1),
                              x[i, j - 1].unsqueeze(-1), ind]
                x[i, j] = pyro.sample("x_{}_{}".format(i, j),
                                      dist.Categorical(probs))

    def guide():
        pass

    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))


def test_enum_recycling_reentrant():
    data = (True, False)
    for i in range(5):
        data = (data, data, False)

    @pyro.markov
    def model(data, state=0, address=""):
        if isinstance(data, bool):
            p = pyro.param("p_leaf", torch.ones(10))
            pyro.sample("leaf_{}".format(address),
                        dist.Bernoulli(p[state]),
                        obs=torch.tensor(1. if data else 0.))
        else:
            p = pyro.param("p_branch", torch.ones(10, 10))
            for branch, letter in zip(data, "abcdefg"):
                next_state = pyro.sample("branch_{}".format(address + letter),
                                         dist.Categorical(p[state]),
                                         infer={"enumerate": "parallel"})
                model(branch, next_state, address + letter)

    def guide(data):
        pass

    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0), data=data)


@pytest.mark.parametrize('history', [1, 2])
def test_enum_recycling_reentrant_history(history):
    data = (True, False)
    for i in range(5):
        data = (data, data, False)

    @pyro.markov(history=history)
    def model(data, state=0, address=""):
        if isinstance(data, bool):
            p = pyro.param("p_leaf", torch.ones(10))
            pyro.sample("leaf_{}".format(address),
                        dist.Bernoulli(p[state]),
                        obs=torch.tensor(1. if data else 0.))
        else:
            assert isinstance(data, tuple)
            p = pyro.param("p_branch", torch.ones(10, 10))
            for branch, letter in zip(data, "abcdefg"):
                next_state = pyro.sample("branch_{}".format(address + letter),
                                         dist.Categorical(p[state]),
                                         infer={"enumerate": "parallel"})
                model(branch, next_state, address + letter)

    def guide(data):
        pass

    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0), data=data)


def test_enum_recycling_mutual_recursion():
    data = (True, False)
    for i in range(5):
        data = (data, data, False)

    def model_leaf(data, state=0, address=""):
        p = pyro.param("p_leaf", torch.ones(10))
        pyro.sample("leaf_{}".format(address),
                    dist.Bernoulli(p[state]),
                    obs=torch.tensor(1. if data else 0.))

    @pyro.markov
    def model1(data, state=0, address=""):
        if isinstance(data, bool):
            model_leaf(data, state, address)
        else:
            p = pyro.param("p_branch", torch.ones(10, 10))
            for branch, letter in zip(data, "abcdefg"):
                next_state = pyro.sample("branch_{}".format(address + letter),
                                         dist.Categorical(p[state]),
                                         infer={"enumerate": "parallel"})
                model2(branch, next_state, address + letter)

    @pyro.markov
    def model2(data, state=0, address=""):
        if isinstance(data, bool):
            model_leaf(data, state, address)
        else:
            p = pyro.param("p_branch", torch.ones(10, 10))
            for branch, letter in zip(data, "abcdefg"):
                next_state = pyro.sample("branch_{}".format(address + letter),
                                         dist.Categorical(p[state]),
                                         infer={"enumerate": "parallel"})
                model1(branch, next_state, address + letter)

    def guide(data):
        pass

    assert_ok(model1, guide, TraceEnum_ELBO(max_plate_nesting=0), data=data)


def test_enum_recycling_interleave():

    def model():
        with pyro.markov() as m:
            with pyro.markov():
                with m:  # error here
                    pyro.sample("x", dist.Categorical(torch.ones(4)),
                                infer={"enumerate": "parallel"})

    def guide():
        pass

    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0, strict_enumeration_warning=False))


def test_enum_recycling_plate():

    @config_enumerate
    def model():
        p = pyro.param("p", torch.ones(3, 3))
        q = pyro.param("q", torch.tensor([0.5, 0.5]))
        plate_x = pyro.plate("plate_x", 2, dim=-1)
        plate_y = pyro.plate("plate_y", 3, dim=-1)
        plate_z = pyro.plate("plate_z", 4, dim=-2)

        a = pyro.sample("a", dist.Bernoulli(q[0])).long()
        w = 0
        for i in pyro.markov(range(5)):
            w = pyro.sample("w_{}".format(i), dist.Categorical(p[w]))

        with plate_x:
            b = pyro.sample("b", dist.Bernoulli(q[a])).long()
            x = 0
            for i in pyro.markov(range(6)):
                x = pyro.sample("x_{}".format(i), dist.Categorical(p[x]))

        with plate_y:
            c = pyro.sample("c", dist.Bernoulli(q[a])).long()
            y = 0
            for i in pyro.markov(range(7)):
                y = pyro.sample("y_{}".format(i), dist.Categorical(p[y]))

        with plate_z:
            d = pyro.sample("d", dist.Bernoulli(q[a])).long()
            z = 0
            for i in pyro.markov(range(8)):
                z = pyro.sample("z_{}".format(i), dist.Categorical(p[z]))

        with plate_x, plate_z:
            e = pyro.sample("e", dist.Bernoulli(q[b])).long()
            xz = 0
            for i in pyro.markov(range(9)):
                xz = pyro.sample("xz_{}".format(i), dist.Categorical(p[xz]))

        return a, b, c, d, e

    def guide():
        pass

    assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=2))


@pytest.mark.parametrize('history', [0, 1, 2, 3])
def test_markov_history(history):

    @config_enumerate
    def model():
        p = pyro.param("p", 0.25 * torch.ones(2, 2))
        q = pyro.param("q", 0.25 * torch.ones(2))
        x_prev = torch.tensor(0)
        x_curr = torch.tensor(0)
        for t in pyro.markov(range(10), history=history):
            probs = p[x_prev, x_curr]
            x_prev, x_curr = x_curr, pyro.sample("x_{}".format(t), dist.Bernoulli(probs)).long()
            pyro.sample("y_{}".format(t), dist.Bernoulli(q[x_curr]),
                        obs=torch.tensor(0.))

    def guide():
        pass

    if history < 2:
        assert_error(model, guide, TraceEnum_ELBO(max_plate_nesting=0),
                     match="Enumeration dim conflict")
    else:
        assert_ok(model, guide, TraceEnum_ELBO(max_plate_nesting=0))


def test_mean_field_ok():

    def model():
        x = pyro.sample("x", dist.Normal(0., 1.))
        pyro.sample("y", dist.Normal(x, 1.))

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        x = pyro.sample("x", dist.Normal(loc, 1.))
        pyro.sample("y", dist.Normal(x, 1.))

    assert_ok(model, guide, TraceMeanField_ELBO())


@pytest.mark.parametrize('mask', [True, False])
def test_mean_field_mask_ok(mask):

    def model():
        x = pyro.sample("x", dist.Normal(0., 1.).mask(mask))
        pyro.sample("y", dist.Normal(x, 1.))

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        x = pyro.sample("x", dist.Normal(loc, 1.).mask(mask))
        pyro.sample("y", dist.Normal(x, 1.))

    assert_ok(model, guide, TraceMeanField_ELBO())


def test_mean_field_warn():

    def model():
        x = pyro.sample("x", dist.Normal(0., 1.))
        pyro.sample("y", dist.Normal(x, 1.))

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        y = pyro.sample("y", dist.Normal(loc, 1.))
        pyro.sample("x", dist.Normal(y, 1.))

    assert_warning(model, guide, TraceMeanField_ELBO())


def test_tail_adaptive_ok():

    def plateless_model():
        pyro.sample("x", dist.Normal(0., 1.))

    def plate_model():
        x = pyro.sample("x", dist.Normal(0., 1.))
        with pyro.plate('observe_data'):
            pyro.sample('obs', dist.Normal(x, 1.0), obs=torch.arange(5).type_as(x))

    def rep_guide():
        pyro.sample("x", dist.Normal(0., 2.))

    assert_ok(plateless_model, rep_guide, TraceTailAdaptive_ELBO(vectorize_particles=True, num_particles=2))
    assert_ok(plate_model, rep_guide, TraceTailAdaptive_ELBO(vectorize_particles=True, num_particles=2))


def test_tail_adaptive_error():

    def plateless_model():
        pyro.sample("x", dist.Normal(0., 1.))

    def rep_guide():
        pyro.sample("x", dist.Normal(0., 2.))

    def nonrep_guide():
        pyro.sample("x", fakes.NonreparameterizedNormal(0., 2.))

    assert_error(plateless_model, rep_guide, TraceTailAdaptive_ELBO(vectorize_particles=False, num_particles=2))
    assert_error(plateless_model, nonrep_guide, TraceTailAdaptive_ELBO(vectorize_particles=True, num_particles=2))


def test_tail_adaptive_warning():

    def plateless_model():
        pyro.sample("x", dist.Normal(0., 1.))

    def rep_guide():
        pyro.sample("x", dist.Normal(0., 2.))

    assert_warning(plateless_model, rep_guide, TraceTailAdaptive_ELBO(vectorize_particles=True, num_particles=1))


@pytest.mark.parametrize("Elbo", [
    Trace_ELBO,
    TraceMeanField_ELBO,
    EnergyDistance_prior,
    EnergyDistance_noprior,
])
def test_reparam_ok(Elbo):

    def model():
        x = pyro.sample("x", dist.Normal(0., 1.))
        pyro.sample("y", dist.Normal(x, 1.), obs=torch.tensor(0.))

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        pyro.sample("x", dist.Normal(loc, 1.))

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("mask", [True, False, torch.tensor(True), torch.tensor(False)])
@pytest.mark.parametrize("Elbo", [
    Trace_ELBO,
    TraceMeanField_ELBO,
    EnergyDistance_prior,
    EnergyDistance_noprior,
])
def test_reparam_mask_ok(Elbo, mask):

    def model():
        x = pyro.sample("x", dist.Normal(0., 1.))
        with poutine.mask(mask=mask):
            pyro.sample("y", dist.Normal(x, 1.), obs=torch.tensor(0.))

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        pyro.sample("x", dist.Normal(loc, 1.))

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("mask", [
    True,
    False,
    torch.tensor(True),
    torch.tensor(False),
    torch.tensor([False, True]),
])
@pytest.mark.parametrize("Elbo", [
    Trace_ELBO,
    TraceMeanField_ELBO,
    EnergyDistance_prior,
    EnergyDistance_noprior,
])
def test_reparam_mask_plate_ok(Elbo, mask):
    data = torch.randn(2, 3).exp()
    data /= data.sum(-1, keepdim=True)

    def model():
        c = pyro.sample("c", dist.LogNormal(0., 1.).expand([3]).to_event(1))
        with pyro.plate("data", len(data)), poutine.mask(mask=mask):
            pyro.sample("obs", dist.Dirichlet(c), obs=data)

    def guide():
        loc = pyro.param("loc", torch.zeros(3))
        scale = pyro.param("scale", torch.ones(3),
                           constraint=constraints.positive)
        pyro.sample("c", dist.LogNormal(loc, scale).to_event(1))

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("scale", [1, 0.1, torch.tensor(0.5)])
@pytest.mark.parametrize("Elbo", [
    Trace_ELBO,
    TraceMeanField_ELBO,
    EnergyDistance_prior,
    EnergyDistance_noprior,
])
def test_reparam_scale_ok(Elbo, scale):

    def model():
        x = pyro.sample("x", dist.Normal(0., 1.))
        with poutine.scale(scale=scale):
            pyro.sample("y", dist.Normal(x, 1.), obs=torch.tensor(0.))

    def guide():
        loc = pyro.param("loc", torch.tensor(0.))
        pyro.sample("x", dist.Normal(loc, 1.))

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("scale", [
    1,
    0.1,
    torch.tensor(0.5),
    torch.tensor([0.1, 0.9]),
])
@pytest.mark.parametrize("Elbo", [
    Trace_ELBO,
    TraceMeanField_ELBO,
    EnergyDistance_prior,
    EnergyDistance_noprior,
])
def test_reparam_scale_plate_ok(Elbo, scale):
    data = torch.randn(2, 3).exp()
    data /= data.sum(-1, keepdim=True)

    def model():
        c = pyro.sample("c", dist.LogNormal(0., 1.).expand([3]).to_event(1))
        with pyro.plate("data", len(data)), poutine.scale(scale=scale):
            pyro.sample("obs", dist.Dirichlet(c), obs=data)

    def guide():
        loc = pyro.param("loc", torch.zeros(3))
        scale = pyro.param("scale", torch.ones(3),
                           constraint=constraints.positive)
        pyro.sample("c", dist.LogNormal(loc, scale).to_event(1))

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [
    EnergyDistance_prior,
    EnergyDistance_noprior,
])
def test_no_log_prob_ok(Elbo):

    def model(data):
        loc = pyro.sample("loc", dist.Normal(0, 1))
        scale = pyro.sample("scale", dist.LogNormal(0, 1))
        with pyro.plate("data", len(data)):
            pyro.sample("obs", dist.Stable(1.5, 0.5, scale, loc),
                        obs=data)

    def guide(data):
        map_loc = pyro.param("map_loc", torch.tensor(0.))
        map_scale = pyro.param("map_scale", torch.tensor(1.),
                               constraint=constraints.positive)
        pyro.sample("loc", dist.Delta(map_loc))
        pyro.sample("scale", dist.Delta(map_scale))

    data = torch.randn(10)
    assert_ok(model, guide, Elbo(), data=data)


def test_reparam_stable():

    @poutine.reparam(config={"z": LatentStableReparam()})
    def model():
        stability = pyro.sample("stability", dist.Uniform(0., 2.))
        skew = pyro.sample("skew", dist.Uniform(-1., 1.))
        y = pyro.sample("z", dist.Stable(stability, skew))
        pyro.sample("x", dist.Poisson(y.abs()), obs=torch.tensor(1.))

    def guide():
        pyro.sample("stability", dist.Delta(torch.tensor(1.5)))
        pyro.sample("skew", dist.Delta(torch.tensor(0.)))
        pyro.sample("z_uniform", dist.Delta(torch.tensor(0.1)))
        pyro.sample("z_exponential", dist.Delta(torch.tensor(1.)))

    assert_ok(model, guide, Trace_ELBO())
