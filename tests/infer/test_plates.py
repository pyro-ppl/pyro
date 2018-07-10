from __future__ import absolute_import, division, print_function

import logging
import warnings

import pytest
import torch

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceGraph_ELBO, config_enumerate
from pyro.optim import Adam
# import pyro.poutine as poutine
from pyro.poutine.plate_messenger import PlateMessenger

logger = logging.getLogger(__name__)

# This file tests a variety of model,guide pairs with valid and invalid structure.


def assert_ok(model, guide, elbo):
    """
    Assert that inference works without warnings or errors.
    """
    pyro.clear_param_store()
    inference = SVI(model, guide, Adam({"lr": 1e-6}), elbo)
    inference.step()


def assert_error(model, guide, elbo):
    """
    Assert that inference fails with an error.
    """
    pyro.clear_param_store()
    inference = SVI(model,  guide, Adam({"lr": 1e-6}), elbo)
    with pytest.raises((NotImplementedError, UserWarning, KeyError, ValueError, RuntimeError)):
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


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nonnested_iarange_iarange_ok(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with PlateMessenger("iarange_0", 10) as ind1:
            pyro.sample("x0", dist.Bernoulli(p).expand_by([len(ind1)]))
        with PlateMessenger("iarange_1", 11) as ind2:
            pyro.sample("x1", dist.Bernoulli(p).expand_by([len(ind2)]))

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_ok(model, guide, Elbo())


def test_three_indep_iarange_at_different_depths_ok():
    """
      /\
     /\ ia
    ia ia
    """
    def model():
        p = torch.tensor(0.5)
        inner_iarange = PlateMessenger("iarange1", 10)
        for i in pyro.irange("irange0", 2):
            pyro.sample("x_%d" % i, dist.Bernoulli(p))
            if i == 0:
                for j in pyro.irange("irange1", 2):
                    with inner_iarange as ind:
                        pyro.sample("y_%d" % j, dist.Bernoulli(p).expand_by([len(ind)]))
            elif i == 1:
                with inner_iarange as ind:
                    pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        inner_iarange = PlateMessenger("iarange1", 10)
        for i in pyro.irange("irange0", 2):
            pyro.sample("x_%d" % i, dist.Bernoulli(p))
            if i == 0:
                for j in pyro.irange("irange1", 2):
                    with inner_iarange as ind:
                        pyro.sample("y_%d" % j, dist.Bernoulli(p).expand_by([len(ind)]))
            elif i == 1:
                with inner_iarange as ind:
                    pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind)]))

    assert_ok(model, guide, TraceGraph_ELBO())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_iarange_no_size_ok(Elbo):

    def model():
        p = torch.tensor(0.5)
        with PlateMessenger("iarange"):
            pyro.sample("x", dist.Bernoulli(p).expand_by([10]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with PlateMessenger("iarange"):
            pyro.sample("x", dist.Bernoulli(p).expand_by([10]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_iarange_iarange_ok(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with PlateMessenger("iarange_outer", 10) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer)]))
            with PlateMessenger("iarange_inner", 11) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner), len(ind_outer)]))

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_iarange_reuse_ok(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        iarange_outer = PlateMessenger("iarange_outer", 10, dim=-1)
        iarange_inner = PlateMessenger("iarange_inner", 11, dim=-2)
        with iarange_outer as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer)]))
        with iarange_inner as ind_inner:
            pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner), 1]))
        with iarange_outer as ind_outer, iarange_inner as ind_inner:
            pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_inner), len(ind_outer)]))

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_ok(model, guide, Elbo())
