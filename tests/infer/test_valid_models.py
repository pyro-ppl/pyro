from __future__ import absolute_import, division, print_function

import logging
import warnings

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceGraph_ELBO, config_enumerate
from pyro.optim import Adam

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
@pytest.mark.parametrize("strict_enumeration_warning", [True, False])
def test_nonempty_model_empty_guide_ok(Elbo, strict_enumeration_warning):

    def model():
        loc = torch.tensor([0.0, 0.0])
        scale = torch.tensor([1.0, 1.0])
        pyro.sample("x", dist.Normal(loc, scale).independent(1), obs=loc)

    def guide():
        pass

    elbo = Elbo(strict_enumeration_warning=strict_enumeration_warning)
    if strict_enumeration_warning and Elbo is TraceEnum_ELBO:
        assert_warning(model, guide, elbo)
    else:
        assert_ok(model, guide, elbo)


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
@pytest.mark.parametrize("strict_enumeration_warning", [True, False])
def test_empty_model_empty_guide_ok(Elbo, strict_enumeration_warning):

    def model():
        pass

    def guide():
        pass

    elbo = Elbo(strict_enumeration_warning=strict_enumeration_warning)
    if strict_enumeration_warning and Elbo is TraceEnum_ELBO:
        assert_warning(model, guide, elbo)
    else:
        assert_ok(model, guide, elbo)


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_variable_clash_in_model_error(Elbo):

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p))
        pyro.sample("x", dist.Bernoulli(p))  # Should error here.

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))

    assert_error(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_model_guide_dim_mismatch_error(Elbo):

    def model():
        loc = torch.zeros(2)
        scale = torch.zeros(2)
        pyro.sample("x", dist.Normal(loc, scale))

    def guide():
        loc = pyro.param("loc", torch.zeros(2, 1, requires_grad=True))
        scale = pyro.param("scale", torch.zeros(2, 1, requires_grad=True))
        pyro.sample("x", dist.Normal(loc, scale))

    assert_error(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_model_guide_shape_mismatch_error(Elbo):

    def model():
        loc = torch.zeros(1, 2)
        scale = torch.zeros(1, 2)
        pyro.sample("x", dist.Normal(loc, scale))

    def guide():
        loc = pyro.param("loc", torch.zeros(2, 1, requires_grad=True))
        scale = pyro.param("scale", torch.zeros(2, 1, requires_grad=True))
        pyro.sample("x", dist.Normal(loc, scale))

    assert_error(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_variable_clash_in_guide_error(Elbo):

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))
        pyro.sample("x", dist.Bernoulli(p))  # Should error here.

    assert_error(model, guide, Elbo())


@pytest.mark.parametrize("subsample_size", [None, 2], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_irange_ok(subsample_size, Elbo):

    def model():
        p = torch.tensor(0.5)
        for i in pyro.irange("irange", 4, subsample_size):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        for i in pyro.irange("irange", 4, subsample_size):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_irange_variable_clash_error(Elbo):

    def model():
        p = torch.tensor(0.5)
        for i in pyro.irange("irange", 2):
            # Each loop iteration should give the sample site a different name.
            pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        for i in pyro.irange("irange", 2):
            # Each loop iteration should give the sample site a different name.
            pyro.sample("x", dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)

    assert_error(model, guide, Elbo())


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_iarange_ok(subsample_size, Elbo):

    def model():
        p = torch.tensor(0.5)
        with pyro.iarange("iarange", 10, subsample_size) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.iarange("iarange", 10, subsample_size) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind)]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_iarange_no_size_ok(Elbo):

    def model():
        p = torch.tensor(0.5)
        with pyro.iarange("iarange"):
            pyro.sample("x", dist.Bernoulli(p).expand_by([10]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.iarange("iarange"):
            pyro.sample("x", dist.Bernoulli(p).expand_by([10]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("subsample_size", [None, 2], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_irange_irange_ok(subsample_size, Elbo):

    def model():
        p = torch.tensor(0.5)
        outer_irange = pyro.irange("irange_0", 3, subsample_size)
        inner_irange = pyro.irange("irange_1", 3, subsample_size)
        for i in outer_irange:
            for j in inner_irange:
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        outer_irange = pyro.irange("irange_0", 3, subsample_size)
        inner_irange = pyro.irange("irange_1", 3, subsample_size)
        for i in outer_irange:
            for j in inner_irange:
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide, "parallel")

    assert_ok(model, guide, Elbo(max_iarange_nesting=0))


@pytest.mark.parametrize("subsample_size", [None, 2], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_irange_irange_swap_ok(subsample_size, Elbo):

    def model():
        p = torch.tensor(0.5)
        outer_irange = pyro.irange("irange_0", 3, subsample_size)
        inner_irange = pyro.irange("irange_1", 3, subsample_size)
        for i in outer_irange:
            for j in inner_irange:
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        outer_irange = pyro.irange("irange_0", 3, subsample_size)
        inner_irange = pyro.irange("irange_1", 3, subsample_size)
        for j in inner_irange:
            for i in outer_irange:
                pyro.sample("x_{}_{}".format(i, j), dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide, "parallel")

    assert_ok(model, guide, Elbo(max_iarange_nesting=0))


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_irange_in_model_not_guide_ok(subsample_size, Elbo):

    def model():
        p = torch.tensor(0.5)
        for i in pyro.irange("irange", 10, subsample_size):
            pass
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
@pytest.mark.parametrize("is_validate", [True, False])
def test_irange_in_guide_not_model_error(subsample_size, Elbo, is_validate):

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        for i in pyro.irange("irange", 10, subsample_size):
            pass
        pyro.sample("x", dist.Bernoulli(p))

    with pyro.validation_enabled(is_validate):
        if is_validate:
            assert_error(model, guide, Elbo())
        else:
            assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_iarange_broadcast_error(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.iarange("iarange", 10, 5):
            pyro.sample("x", dist.Bernoulli(p).expand_by([2]))

    assert_error(model, model, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_iarange_irange_ok(Elbo):

    def model():
        p = torch.tensor(0.5)
        with pyro.iarange("iarange", 3, 2) as ind:
            for i in pyro.irange("irange", 3, 2):
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.iarange("iarange", 3, 2) as ind:
            for i in pyro.irange("irange", 3, 2):
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_irange_iarange_ok(Elbo):

    def model():
        p = torch.tensor(0.5)
        inner_iarange = pyro.iarange("iarange", 3, 2)
        for i in pyro.irange("irange", 3, 2):
            with inner_iarange as ind:
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        inner_iarange = pyro.iarange("iarange", 3, 2)
        for i in pyro.irange("irange", 3, 2):
            with inner_iarange as ind:
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_iarange_iarange_ok(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer)]))
            with pyro.iarange("iarange_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner), len(ind_outer)]))

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_iarange_reuse_ok(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        iarange_outer = pyro.iarange("iarange_outer", 10, 5, dim=-1)
        iarange_inner = pyro.iarange("iarange_inner", 11, 6, dim=-2)
        with iarange_outer as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer)]))
        with iarange_inner as ind_inner:
            pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner), 1]))
        with iarange_outer as ind_outer, iarange_inner as ind_inner:
            pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_inner), len(ind_outer)]))

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_iarange_iarange_dim_error_1(Elbo):

    def model():
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer)]))  # error here
            with pyro.iarange("iarange_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner)]))
                pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_outer), len(ind_inner)]))

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_iarange_iarange_dim_error_2(Elbo):

    def model():
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer), 1]))
            with pyro.iarange("iarange_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_outer)]))  # error here
                pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_outer), len(ind_inner)]))

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_iarange_iarange_dim_error_3(Elbo):

    def model():
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer), 1]))
            with pyro.iarange("iarange_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner)]))
                pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_inner), 1]))  # error here

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_iarange_iarange_dim_error_4(Elbo):

    def model():
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer), 1]))
            with pyro.iarange("iarange_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner)]))
                pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_outer), len(ind_outer)]))  # error here

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nonnested_iarange_iarange_ok(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.iarange("iarange_0", 10, 5) as ind1:
            pyro.sample("x0", dist.Bernoulli(p).expand_by([len(ind1)]))
        with pyro.iarange("iarange_1", 11, 6) as ind2:
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
        inner_iarange = pyro.iarange("iarange1", 10, 5)
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
        inner_iarange = pyro.iarange("iarange1", 10, 5)
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


def test_iarange_wrong_size_error():

    def model():
        p = torch.tensor(0.5)
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([1 + len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([1 + len(ind)]))

    assert_error(model, guide, TraceGraph_ELBO())


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
        assert_warning(model, guide, Elbo(max_iarange_nesting=0))
    else:
        assert_ok(model, guide, Elbo(max_iarange_nesting=0))


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


def test_enum_discrete_irange_single_ok():

    def model():
        p = torch.tensor(0.5)
        for i in pyro.irange("irange", 10, 5):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        for i in pyro.irange("irange", 10, 5):
            pyro.sample("x_{}".format(i), dist.Bernoulli(p))

    assert_ok(model, config_enumerate(guide), TraceEnum_ELBO())


def test_iarange_enum_discrete_batch_ok():

    def model():
        p = torch.tensor(0.5)
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind)]))

    assert_ok(model, config_enumerate(guide), TraceEnum_ELBO())


@pytest.mark.parametrize("strict_enumeration_warning", [False, True])
def test_iarange_enum_discrete_no_discrete_vars_warning(strict_enumeration_warning):

    def model():
        loc = torch.tensor(0.0)
        scale = torch.tensor(1.0)
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Normal(loc, scale).expand_by([len(ind)]))

    @config_enumerate
    def guide():
        loc = pyro.param("loc", torch.tensor(1.0, requires_grad=True))
        scale = pyro.param("scale", torch.tensor(2.0, requires_grad=True))
        with pyro.iarange("iarange", 10, 5) as ind:
            pyro.sample("x", dist.Normal(loc, scale).expand_by([len(ind)]))

    elbo = TraceEnum_ELBO(strict_enumeration_warning=strict_enumeration_warning)
    if strict_enumeration_warning:
        assert_warning(model, guide, elbo)
    else:
        assert_ok(model, guide, elbo)


def test_no_iarange_enum_discrete_batch_error():

    def model():
        p = torch.tensor(0.5)
        pyro.sample("x", dist.Bernoulli(p).expand_by([5]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        pyro.sample("x", dist.Bernoulli(p).expand_by([5]))

    assert_error(model, config_enumerate(guide), TraceEnum_ELBO())


@pytest.mark.parametrize('max_iarange_nesting', [0, 1, 2])
def test_enum_discrete_parallel_ok(max_iarange_nesting):
    iarange_shape = torch.Size([1] * max_iarange_nesting)

    def model():
        p = torch.tensor(0.5)
        x = pyro.sample("x", dist.Bernoulli(p))
        assert x.shape == torch.Size([2]) + iarange_shape

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        x = pyro.sample("x", dist.Bernoulli(p))
        assert x.shape == torch.Size([2]) + iarange_shape

    assert_ok(model, config_enumerate(guide, "parallel"),
              TraceEnum_ELBO(max_iarange_nesting=max_iarange_nesting))


@pytest.mark.parametrize('max_iarange_nesting', [0, 1, 2])
def test_enum_discrete_parallel_nested_ok(max_iarange_nesting):
    iarange_shape = torch.Size([1] * max_iarange_nesting)

    def model():
        p2 = torch.tensor(torch.ones(2) / 2)
        p3 = torch.tensor(torch.ones(3) / 3)
        x2 = pyro.sample("x2", dist.OneHotCategorical(p2))
        x3 = pyro.sample("x3", dist.OneHotCategorical(p3))
        assert x2.shape == torch.Size([2]) + iarange_shape + p2.shape
        assert x3.shape == torch.Size([3, 1]) + iarange_shape + p3.shape

    assert_ok(model, config_enumerate(model, "parallel"),
              TraceEnum_ELBO(max_iarange_nesting=max_iarange_nesting))


@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("enumerate_", ["sequential", "parallel"])
def test_enumerate_parallel_iarange_ok(enumerate_, expand):

    def model():
        p2 = torch.ones(2) / 2
        p34 = torch.ones(3, 4) / 4
        p536 = torch.ones(5, 3, 6) / 6

        x2 = pyro.sample("x2", dist.Categorical(p2))
        with pyro.iarange("outer", 3):
            x34 = pyro.sample("x34", dist.Categorical(p34))
            with pyro.iarange("inner", 5):
                x536 = pyro.sample("x536", dist.Categorical(p536))

        # check shapes
        if enumerate_ == "sequential":
            if expand:
                # All dimensions are iarange dimensions.
                assert x2.shape == torch.Size([])
                assert x34.shape == torch.Size([3])
                assert x536.shape == torch.Size([5, 3])
            else:
                # All dimensions are iarange placeholders.
                assert x2.shape == torch.Size([])
                assert x34.shape == torch.Size([1])
                assert x536.shape == torch.Size([1, 1])
        else:
            if expand:
                # Meaning of dimensions:    [ enum dims | iarange dims ]
                assert x2.shape == torch.Size([        2, 1, 1])  # noqa: E201
                assert x34.shape == torch.Size([    4, 1, 1, 3])  # noqa: E201
                assert x536.shape == torch.Size([6, 1, 1, 5, 3])  # noqa: E201
            else:
                # Meaning of dimensions:    [ enum dims | iarange placeholders ]
                assert x2.shape == torch.Size([        2, 1, 1])  # noqa: E201
                assert x34.shape == torch.Size([    4, 1, 1, 1])  # noqa: E201
                assert x536.shape == torch.Size([6, 1, 1, 1, 1])  # noqa: E201

    elbo = TraceEnum_ELBO(max_iarange_nesting=2)
    assert_ok(model, config_enumerate(model, enumerate_, expand), elbo)


@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
@pytest.mark.parametrize('is_validate', [True, False])
def test_enum_discrete_iarange_dependency_warning(enumerate_, is_validate):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        with pyro.iarange("iarange", 10, 5):
            x = pyro.sample("x", dist.Bernoulli(0.5).expand_by([5]),
                            infer={'enumerate': enumerate_})
        pyro.sample("y", dist.Bernoulli(x.mean()))  # user should move this line up

    with pyro.validation_enabled(is_validate):
        if enumerate_ and is_validate:
            assert_warning(model, model, TraceEnum_ELBO(max_iarange_nesting=1))
        else:
            assert_ok(model, model, TraceEnum_ELBO(max_iarange_nesting=1))


@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
def test_enum_discrete_irange_iarange_dependency_ok(enumerate_):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        inner_iarange = pyro.iarange("iarange", 10, 5)
        for i in pyro.irange("irange", 3):
            pyro.sample("y_{}".format(i), dist.Bernoulli(0.5))
            with inner_iarange:
                pyro.sample("x_{}".format(i), dist.Bernoulli(0.5).expand_by([5]),
                            infer={'enumerate': enumerate_})

    assert_ok(model, model, TraceEnum_ELBO(max_iarange_nesting=1))


@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
@pytest.mark.parametrize('is_validate', [True, False])
def test_enum_discrete_iranges_iarange_dependency_warning(enumerate_, is_validate):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        inner_iarange = pyro.iarange("iarange", 10, 5)

        for i in pyro.irange("irange1", 2):
            with inner_iarange:
                pyro.sample("x_{}".format(i), dist.Bernoulli(0.5).expand_by([5]),
                            infer={'enumerate': enumerate_})

        for i in pyro.irange("irange2", 2):
            pyro.sample("y_{}".format(i), dist.Bernoulli(0.5))

    with pyro.validation_enabled(is_validate):
        if enumerate_ and is_validate:
            assert_warning(model, model, TraceEnum_ELBO(max_iarange_nesting=1))
        else:
            assert_ok(model, model, TraceEnum_ELBO(max_iarange_nesting=1))


@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
def test_enum_discrete_iaranges_dependency_ok(enumerate_):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        x_iarange = pyro.iarange("x_iarange", 10, 5, dim=-1)
        y_iarange = pyro.iarange("y_iarange", 11, 6, dim=-2)
        pyro.sample("a", dist.Bernoulli(0.5))
        with x_iarange:
            pyro.sample("b", dist.Bernoulli(0.5).expand_by([5]))
        with y_iarange:
            # Note that it is difficult to check that c does not depend on b.
            pyro.sample("c", dist.Bernoulli(0.5).expand_by([6, 1]))
        with x_iarange, y_iarange:
            pyro.sample("d", dist.Bernoulli(0.5).expand_by([6, 5]))

    assert_ok(model, model, TraceEnum_ELBO(max_iarange_nesting=2))


@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
def test_enum_discrete_non_enumerated_iarange_ok(enumerate_):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})

        with pyro.iarange("non_enum", 2):
            a = pyro.sample("a", dist.Bernoulli(0.5).expand_by([2]),
                            infer={'enumerate': None})

        p = (1.0 + a.sum(-1)) / (2.0 + a.size(0))  # introduce dependency of b on a

        with pyro.iarange("enum_1", 3):
            pyro.sample("b", dist.Bernoulli(p).expand_by([3]),
                        infer={'enumerate': enumerate_})

    with pyro.validation_enabled():
        assert_ok(model, model, TraceEnum_ELBO(max_iarange_nesting=1))


@pytest.mark.parametrize("times", [1, 2], ids=["applied_once", "applied_twice"])
def test_iarange_shape_broadcasting(times):
    data = torch.ones(1000, 2)

    def model():
        with pyro.iarange("num_particles", 10, dim=-3):
            with pyro.iarange("components", 2, dim=-1):
                p = pyro.sample("p", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
                assert p.shape == torch.Size((10, 1, 2))
            with pyro.iarange("data", data.shape[0], dim=-2):
                pyro.sample("obs", dist.Bernoulli(p), obs=data)

    def guide():
        with pyro.iarange("num_particles", 10, dim=-3):
            with pyro.iarange("components", 2, dim=-1):
                pyro.sample("p", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))

    pyro.clear_param_store()
    for _ in range(times):
        model, guide = poutine.broadcast(model), poutine.broadcast(guide)
    assert_ok(model, guide, Trace_ELBO())


@pytest.mark.parametrize('enumerate_,expand', [
    (None, True),
    ("sequential", True),
    ("sequential", False),
    ("parallel", True),
    ("parallel", False),
])
def test_enum_discrete_iarange_shape_broadcasting_ok(enumerate_, expand):

    @poutine.broadcast
    def model():
        x_iarange = pyro.iarange("x_iarange", 10, 5, dim=-1)
        y_iarange = pyro.iarange("y_iarange", 11, 6, dim=-2)
        with pyro.iarange("num_particles", 50, dim=-3):
            with x_iarange:
                b = pyro.sample("b", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
            with y_iarange:
                c = pyro.sample("c", dist.Bernoulli(0.5))
            with x_iarange, y_iarange:
                d = pyro.sample("d", dist.Bernoulli(b))

        # check shapes
        assert b.shape == (50, 1, 5)
        if enumerate_ == "parallel":
            if expand:
                assert c.shape == (2, 50, 6, 1)  # FIXME actual shape == (2, 1, 1, 1)
                assert d.shape == (2, 1, 50, 6, 5)
            else:
                assert c.shape == (2, 1, 1, 1)
                assert d.shape == (2, 1, 1, 1, 1)
        elif enumerate_ == "sequential":
            if expand:
                assert c.shape == (50, 6, 1)  # FIXME actual shape == ()
                assert d.shape == (50, 6, 5)
            else:
                assert c.shape == (1, 1, 1)  # FIXME actual shape == ()
                assert d.shape == (1, 1, 1)
        else:
            assert c.shape == (50, 6, 1)
            assert d.shape == (50, 6, 5)

    guide = config_enumerate(model, default=enumerate_, expand=expand)
    elbo = TraceEnum_ELBO(max_iarange_nesting=3,
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

    @poutine.broadcast
    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5, dim=-3):
            x = pyro.sample("x", dist.Bernoulli(p))
            with pyro.iarange("iarange_inner_1", 11, 6):
                y = pyro.sample("y", dist.Bernoulli(p))
                # allocated dim is rightmost available, i.e. -1
                with pyro.iarange("iarange_inner_2", 12, 7):
                    z = pyro.sample("z", dist.Bernoulli(p))
                    # allocated dim is next rightmost available, i.e. -2
                    # since dim -3 is already allocated, use dim=-4
                    with pyro.iarange("iarange_inner_3", 13, 8):
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

    guide = config_enumerate(model, expand=expand) if enumerate_ else model
    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo,expand", [
    (Trace_ELBO, False),
    (TraceGraph_ELBO, False),
    (TraceEnum_ELBO, False),
    (TraceEnum_ELBO, True),
])
def test_dim_allocation_error(Elbo, expand):
    enumerate_ = (Elbo is TraceEnum_ELBO)

    @poutine.broadcast
    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.iarange("iarange_outer", 10, 5, dim=-2):
            x = pyro.sample("x", dist.Bernoulli(p))
            # allocated dim is rightmost available, i.e. -1
            with pyro.iarange("iarange_inner_1", 11, 6):
                y = pyro.sample("y", dist.Bernoulli(p))
                # throws an error as dim=-1 is already occupied
                with pyro.iarange("iarange_inner_2", 12, 7, dim=-1):
                    pyro.sample("z", dist.Bernoulli(p))

        # check shapes
        if enumerate_ and not expand:
            assert x.shape == (1, 1)
            assert y.shape == (1, 1)
        else:
            assert x.shape == (5, 1)
            assert y.shape == (5, 6)

    guide = config_enumerate(model, expand=expand) if Elbo is TraceEnum_ELBO else model
    assert_error(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_vectorized_num_particles(Elbo):
    data = torch.ones(1000, 2)

    @poutine.broadcast
    def model():
        with pyro.iarange("components", 2):
            p = pyro.sample("p", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
            assert p.shape == torch.Size((10, 1, 2))
            with pyro.iarange("data", data.shape[0]):
                pyro.sample("obs", dist.Bernoulli(p), obs=data)

    @poutine.broadcast
    def guide():
        with pyro.iarange("components", 2):
            pyro.sample("p", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))

    pyro.clear_param_store()
    guide = config_enumerate(guide) if Elbo is TraceEnum_ELBO else guide
    assert_ok(model, guide, Elbo(num_particles=10,
                                 vectorize_particles=True,
                                 max_iarange_nesting=2,
                                 strict_enumeration_warning=False))


@pytest.mark.parametrize('enumerate_,expand', [
    (None, False),
    ("sequential", False),
    ("sequential", True),
    ("parallel", False),
    ("parallel", True),
])
@pytest.mark.parametrize('num_particles', [1, 50])
def test_enum_discrete_vectorized_num_particles(enumerate_, expand, num_particles):

    @config_enumerate(default=enumerate_, expand=expand)
    def model():
        x_iarange = pyro.iarange("x_iarange", 10, 5, dim=-1)
        y_iarange = pyro.iarange("y_iarange", 11, 6, dim=-2)
        with x_iarange:
            b = pyro.sample("b", dist.Beta(torch.tensor(1.1), torch.tensor(1.1)))
            assert b.shape == torch.Size((num_particles, 1, 5) if num_particles > 1 else (5,))
        with y_iarange:
            c = pyro.sample("c", dist.Bernoulli(0.5))
        with x_iarange, y_iarange:
            d = pyro.sample("d", dist.Bernoulli(b))

        # check shapes
        if num_particles > 1:
            assert b.shape == (num_particles, 1, 5)
            if enumerate_ == "parallel":
                if expand:
                    assert c.shape == (2, num_particles, 6, 1)
                    assert d.shape == (2, 1, num_particles, 6, 5)
                else:
                    assert c.shape == (2, 1, 1, 1)
                    assert d.shape == (2, 1, 1, 1, 1)
            elif enumerate_ == "sequential":
                if expand:
                    assert c.shape == (num_particles, 6, 1)
                    assert d.shape == (num_particles, 6, 5)
                else:
                    assert c.shape == (1, 1, 1)
                    assert d.shape == (1, 1, 1)
        else:
            assert b.shape == (5,)
            if enumerate_ == "parallel":
                if expand:
                    assert c.shape == (2, 6, 1)
                    assert d.shape == (2, 1, 6, 5)
                else:
                    assert c.shape == (2, 1, 1)
                    assert d.shape == (2, 1, 1, 1)
            elif enumerate_ == "sequential":
                if expand:
                    assert c.shape == (6, 1)
                    assert d.shape == (6, 5)
                else:
                    assert c.shape == (1, 1)
                    assert d.shape == (1, 1)

    assert_ok(model, model, TraceEnum_ELBO(max_iarange_nesting=2,
                                           num_particles=num_particles,
                                           vectorize_particles=True,
                                           strict_enumeration_warning=(enumerate_ == "parallel")))
