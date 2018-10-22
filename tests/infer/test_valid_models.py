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

    assert_error(model, guide, Elbo(), match='Multiple sample sites named')


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_model_guide_dim_mismatch_error(Elbo):

    def model():
        loc = torch.zeros(2)
        scale = torch.ones(2)
        pyro.sample("x", dist.Normal(loc, scale).independent(1))

    def guide():
        loc = pyro.param("loc", torch.zeros(2, 1, requires_grad=True))
        scale = pyro.param("scale", torch.ones(2, 1, requires_grad=True))
        pyro.sample("x", dist.Normal(loc, scale).independent(2))

    assert_error(model, guide, Elbo(strict_enumeration_warning=False),
                 match='invalid log_prob shape|Model and guide event_dims disagree')


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_model_guide_shape_mismatch_error(Elbo):

    def model():
        loc = torch.zeros(1, 2)
        scale = torch.ones(1, 2)
        pyro.sample("x", dist.Normal(loc, scale).independent(2))

    def guide():
        loc = pyro.param("loc", torch.zeros(2, 1, requires_grad=True))
        scale = pyro.param("scale", torch.ones(2, 1, requires_grad=True))
        pyro.sample("x", dist.Normal(loc, scale).independent(2))

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

    assert_error(model, guide, Elbo(), match='Multiple sample sites named')


@pytest.mark.parametrize("subsample_size", [None, 5], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
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

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
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

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("max_plate_nesting", [0, float('inf')])
@pytest.mark.parametrize("subsample_size", [None, 2], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_irange_irange_ok(subsample_size, Elbo, max_plate_nesting):

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

    assert_ok(model, guide, Elbo(max_plate_nesting=max_plate_nesting))


@pytest.mark.parametrize("max_plate_nesting", [0, float('inf')])
@pytest.mark.parametrize("subsample_size", [None, 2], ids=["full", "subsample"])
@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_irange_irange_swap_ok(subsample_size, Elbo, max_plate_nesting):

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

    assert_ok(model, guide, Elbo(max_plate_nesting=max_plate_nesting))


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


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_plate_irange_ok(Elbo):

    def model():
        p = torch.tensor(0.5)
        with pyro.plate("plate", 3, 2) as ind:
            for i in pyro.irange("irange", 3, 2):
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        with pyro.plate("plate", 3, 2) as ind:
            for i in pyro.irange("irange", 3, 2):
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_irange_plate_ok(Elbo):

    def model():
        p = torch.tensor(0.5)
        inner_plate = pyro.plate("plate", 3, 2)
        for i in pyro.irange("irange", 3, 2):
            with inner_plate as ind:
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        inner_plate = pyro.plate("plate", 3, 2)
        for i in pyro.irange("irange", 3, 2):
            with inner_plate as ind:
                pyro.sample("x_{}".format(i), dist.Bernoulli(p).expand_by([len(ind)]))

    if Elbo is TraceEnum_ELBO:
        guide = config_enumerate(guide)

    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_plate_plate_ok(Elbo):

    def model():
        p = torch.tensor(0.5, requires_grad=True)
        with pyro.plate("plate_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer)]))
            with pyro.plate("plate_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner), len(ind_outer)]))

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
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

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
    assert_ok(model, guide, Elbo())


@pytest.mark.parametrize("Elbo", [Trace_ELBO, TraceGraph_ELBO, TraceEnum_ELBO])
def test_nested_plate_plate_dim_error_1(Elbo):

    def model():
        p = torch.tensor([0.5], requires_grad=True)
        with pyro.plate("plate_outer", 10, 5) as ind_outer:
            pyro.sample("x", dist.Bernoulli(p).expand_by([len(ind_outer)]))  # error here
            with pyro.plate("plate_inner", 11, 6) as ind_inner:
                pyro.sample("y", dist.Bernoulli(p).expand_by([len(ind_inner)]))
                pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind_outer), len(ind_inner)]))

    guide = config_enumerate(model) if Elbo is TraceEnum_ELBO else model
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
    """
      /\
     /\ ia
    ia ia
    """
    def model():
        p = torch.tensor(0.5)
        inner_plate = pyro.plate("plate1", 10, 5)
        for i in pyro.irange("irange0", 2):
            pyro.sample("x_%d" % i, dist.Bernoulli(p))
            if i == 0:
                for j in pyro.irange("irange1", 2):
                    with inner_plate as ind:
                        pyro.sample("y_%d" % j, dist.Bernoulli(p).expand_by([len(ind)]))
            elif i == 1:
                with inner_plate as ind:
                    pyro.sample("z", dist.Bernoulli(p).expand_by([len(ind)]))

    def guide():
        p = pyro.param("p", torch.tensor(0.5, requires_grad=True))
        inner_plate = pyro.plate("plate1", 10, 5)
        for i in pyro.irange("irange0", 2):
            pyro.sample("x_%d" % i, dist.Bernoulli(p))
            if i == 0:
                for j in pyro.irange("irange1", 2):
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

    @config_enumerate
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
def test_enum_discrete_irange_plate_dependency_ok(enumerate_, max_plate_nesting):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        inner_plate = pyro.plate("plate", 10, 5)
        for i in pyro.irange("irange", 3):
            pyro.sample("y_{}".format(i), dist.Bernoulli(0.5))
            with inner_plate:
                pyro.sample("x_{}".format(i), dist.Bernoulli(0.5).expand_by([5]),
                            infer={'enumerate': enumerate_})

    assert_ok(model, model, TraceEnum_ELBO(max_plate_nesting=max_plate_nesting))


@pytest.mark.parametrize('max_plate_nesting', [1, float('inf')])
@pytest.mark.parametrize('enumerate_', [None, "sequential", "parallel"])
@pytest.mark.parametrize('is_validate', [True, False])
def test_enum_discrete_iranges_plate_dependency_warning(enumerate_, is_validate, max_plate_nesting):

    def model():
        pyro.sample("w", dist.Bernoulli(0.5), infer={'enumerate': 'parallel'})
        inner_plate = pyro.plate("plate", 10, 5)

        for i in pyro.irange("irange1", 2):
            with inner_plate:
                pyro.sample("x_{}".format(i), dist.Bernoulli(0.5).expand_by([5]),
                            infer={'enumerate': enumerate_})

        for i in pyro.irange("irange2", 2):
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
            if num_samples:
                assert b.shape == (num_samples, 50, 1, 5)
                assert c.shape == (num_samples, 1, 50, 6, 1)
                assert d.shape == (num_samples, 1, num_samples, 50, 6, 5)
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

    guide = config_enumerate(model, expand=expand) if enumerate_ else model
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

    @config_enumerate(default="parallel")
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

    @config_enumerate(default="parallel")
    def model():
        p = pyro.param("p", torch.tensor([0.2, 0.8]))
        x = pyro.sample("x", dist.Bernoulli(0.3)).long()
        with poutine.scale(scale=2.):
            pyro.sample("y", dist.Bernoulli(p[x]), obs=torch.tensor(0.))

    def guide():
        pass

    assert_error(model, guide, TraceEnum_ELBO(max_plate_nesting=0),
                 match='Expected all enumerated sample sites to share a common poutine.scale')


def test_enum_in_model_diamond_error():
    data = torch.tensor([[0, 1], [0, 0]])

    @config_enumerate(default="parallel")
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
        d_ind = torch.arange(2, dtype=torch.long)
        a = pyro.sample("a", dist.Categorical(probs_a))
        with b_axis:
            b = pyro.sample("b", dist.Categorical(probs_b[a]))
        with c_axis:
            c = pyro.sample("c", dist.Categorical(probs_c[a]))
        with b_axis, c_axis:
            pyro.sample("d",
                        dist.Categorical(probs_d[b.unsqueeze(-1), c.unsqueeze(-1), d_ind]),
                        obs=data)

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
                if num_samples:
                    assert b.shape == (num_samples, num_particles, 1, 5)
                    assert c.shape == (num_samples, 1, num_particles, 6, 1)
                    assert d.shape == (num_samples, 1, num_samples, num_particles, 6, 5)
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
                if num_samples:
                    assert b.shape == (num_samples, 1, 5,)
                    assert c.shape == (num_samples, 1, 6, 1)
                    assert d.shape == (num_samples, 1, num_samples, 6, 5)
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
