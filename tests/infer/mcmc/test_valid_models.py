import logging

import torch

import pytest
from torch.nn.functional import sigmoid

import pyro
import pyro.distributions as dist
from pyro.infer import config_enumerate
from pyro.infer.mcmc import MCMC, HMC, NUTS
import pyro.poutine as poutine
from pyro.infer.mcmc.util import EnumTraceProbEvaluator
from pyro.primitives import _Subsample
from tests.common import assert_equal

logger = logging.getLogger(__name__)


def assert_ok(mcmc_kernel):
    """
    Assert that inference works without warnings or errors.
    """
    MCMC(mcmc_kernel, num_samples=10, warmup_steps=10).run()


def assert_error(mcmc_kernel):
    """
    Assert that inference fails with an error.
    """
    with pytest.raises(ValueError):
        MCMC(mcmc_kernel, num_samples=10, warmup_steps=10).run()


def print_debug_info(model_trace):
    model_trace.compute_log_prob()
    for name, site in model_trace.nodes.items():
        if site["type"] == "sample" and not isinstance(site["fn"], _Subsample):
            logger.debug("prob( {} ):\n {}".format(name, site["log_prob"].exp()))


@pytest.mark.parametrize("kernel, kwargs", [
    (HMC, {"adapt_step_size": True, "num_steps": 3}),
    (NUTS, {"adapt_step_size": True}),
])
def test_model_error_stray_batch_dims(kernel, kwargs):

    @poutine.broadcast
    def gmm():
        data = torch.tensor([0., 0., 3., 3., 3., 5., 5.])
        mix_proportions = pyro.sample("phi", dist.Dirichlet(torch.ones(3)))
        cluster_means = pyro.sample("cluster_means", dist.Normal(torch.arange(3), 1.))
        with pyro.iarange("data", data.shape[0]):
            assignments = pyro.sample("assignments", dist.Categorical(mix_proportions))
            pyro.sample("obs", dist.Normal(cluster_means[assignments], 1.), obs=data)
        return cluster_means

    mcmc_kernel = kernel(gmm, **kwargs)
    # Error due to non finite value for `max_iarange_nesting`.
    assert_error(mcmc_kernel)
    # Error due to batch dims not inside iarange.
    mcmc_kernel = kernel(gmm, max_iarange_nesting=1, **kwargs)
    assert_error(mcmc_kernel)


@pytest.mark.parametrize("kernel, kwargs", [
    (HMC, {"adapt_step_size": True, "num_steps": 3}),
    (NUTS, {"adapt_step_size": True}),
])
def test_model_error_enum_dim_clash(kernel, kwargs):

    @poutine.broadcast
    def gmm():
        data = torch.tensor([0., 0., 3., 3., 3., 5., 5.])
        with pyro.iarange("num_clusters", 3):
            mix_proportions = pyro.sample("phi", dist.Dirichlet(torch.tensor(1.)))
            cluster_means = pyro.sample("cluster_means", dist.Normal(torch.arange(3), 1.))
        with pyro.iarange("data", data.shape[0]):
            assignments = pyro.sample("assignments", dist.Categorical(mix_proportions))
            pyro.sample("obs", dist.Normal(cluster_means[assignments], 1.), obs=data)
        return cluster_means

    mcmc_kernel = kernel(gmm, max_iarange_nesting=0, **kwargs)
    assert_error(mcmc_kernel)


def test_all_discrete_sites_log_prob():
    p = 0.3

    @poutine.enum(first_available_dim=2)
    @config_enumerate(default="parallel")
    @poutine.broadcast
    def model():
        d = dist.Bernoulli(p)
        context1 = pyro.iarange("outer", 2, dim=-1)
        context2 = pyro.iarange("inner", 1, dim=-2)
        pyro.sample("w", d)
        with context1:
            pyro.sample("x", d)
        with context2:
            pyro.sample("y", d)
        with context1, context2:
            pyro.sample("z", d)

    model_trace = poutine.trace(model).get_trace()
    print_debug_info(model_trace)
    trace_prob_evaluator = EnumTraceProbEvaluator(model_trace, True, 2)
    assert_equal(trace_prob_evaluator.log_prob(), torch.tensor(0.))


def test_discrete_sites_correctly_enumerated():
    p = 0.3

    @poutine.enum(first_available_dim=2)
    @config_enumerate(default="parallel")
    @poutine.condition(data={"b": torch.tensor(0.4), "c": torch.tensor(0.4)})
    @poutine.broadcast
    def model():
        d = dist.Bernoulli(p)
        context1 = pyro.iarange("outer", 3, dim=-1)
        context2 = pyro.iarange("inner", 2, dim=-2)
        pyro.sample("w", d)
        pyro.sample("b", dist.Beta(1.1, 1.1))
        with context1:
            pyro.sample("x", d)
        with context2:
            pyro.sample("c", dist.Beta(1.1, 1.1))
            pyro.sample("y", d)
        with context1, context2:
            pyro.sample("z", d)

    model_trace = poutine.trace(model).get_trace()
    print_debug_info(model_trace)
    trace_prob_evaluator = EnumTraceProbEvaluator(model_trace, True, 2)
    assert_equal(trace_prob_evaluator.log_prob(), torch.tensor(0.16196))  # p_beta(0.3)^3


@pytest.mark.parametrize("data, expected_log_prob", [
    (torch.tensor([1.]), torch.tensor(-1.3434)),
    (torch.tensor([0.]), torch.tensor(-1.4189)),
    (torch.tensor([1., 0., 0.]), torch.tensor(-4.1813)),
])
def test_enum_log_prob_continuous_observed(data, expected_log_prob):

    @poutine.enum(first_available_dim=1)
    @config_enumerate(default="parallel")
    @poutine.condition(data={"p": torch.tensor(0.4)})
    @poutine.broadcast
    def model(data):
        p = pyro.sample("p", dist.Uniform(0., 1.))
        y = pyro.sample("y", dist.Bernoulli(p))
        q = 0.5 + 0.25 * y
        with pyro.iarange("data", len(data)):
            z = pyro.sample("z", dist.Bernoulli(q))
            mean = 2 * z - 1
            pyro.sample("obs", dist.Normal(mean, 1.), obs=data)

    model_trace = poutine.trace(model).get_trace(data)
    print_debug_info(model_trace)
    trace_prob_evaluator = EnumTraceProbEvaluator(model_trace, True, 1)
    assert_equal(trace_prob_evaluator.log_prob(),
                 expected_log_prob,
                 prec=1e-3)


@pytest.mark.parametrize("data, expected_log_prob", [
    (torch.tensor([1.]), torch.tensor(-3.5237)),
    (torch.tensor([0.]), torch.tensor(-3.7091)),
    (torch.tensor([1., 1.]), torch.tensor(-7.0474)),
    (torch.tensor([1., 0., 0.]), torch.tensor(-10.9418)),
])
def test_enum_log_prob_continuous_sampled(data, expected_log_prob):

    @poutine.enum(first_available_dim=1)
    @config_enumerate(default="parallel")
    @poutine.condition(data={"p": torch.tensor(0.4),
                             "n": torch.tensor([[1.], [-1.]])})
    @poutine.broadcast
    def model(data):
        p = pyro.sample("p", dist.Uniform(0., 1.))
        y = pyro.sample("y", dist.Bernoulli(p))
        mean = 2 * y - 1
        n = pyro.sample("n", dist.Normal(mean, 1.))
        with pyro.iarange("data", len(data)):
            pyro.sample("obs", dist.Bernoulli(sigmoid(n)), obs=data)

    model_trace = poutine.trace(model).get_trace(data)
    print_debug_info(model_trace)
    trace_prob_evaluator = EnumTraceProbEvaluator(model_trace, True, 1)
    assert_equal(trace_prob_evaluator.log_prob(),
                 expected_log_prob,
                 prec=1e-3)


@pytest.mark.parametrize("data, expected_log_prob", [
    (torch.tensor([1.]), torch.tensor(-0.5798)),
    (torch.tensor([1., 1.]), torch.tensor(-1.1596)),
    (torch.tensor([1., 0., 0.]), torch.tensor(-2.2218)),
])
def test_enum_log_prob_discrete_observed(data, expected_log_prob):

    @poutine.enum(first_available_dim=1)
    @config_enumerate(default="parallel")
    @poutine.condition(data={"p": torch.tensor(0.4)})
    @poutine.broadcast
    def model(data):
        p = pyro.sample("p", dist.Uniform(0., 1.))
        y = pyro.sample("y", dist.Bernoulli(p))
        q = 0.25 * y + 0.5
        with pyro.iarange("data", len(data)):
            z = pyro.sample("z", dist.Bernoulli(q))
            p = 0.6 * z + 0.2
            pyro.sample("obs", dist.Bernoulli(p), obs=data)

    model_trace = poutine.trace(model).get_trace(data)
    print_debug_info(model_trace)
    trace_prob_evaluator = EnumTraceProbEvaluator(model_trace, True, 1)
    assert_equal(trace_prob_evaluator.log_prob(),
                 expected_log_prob,
                 prec=1e-3)


@pytest.mark.parametrize("data, expected_log_prob", [
    (torch.tensor([1.]), torch.tensor(-2.7622)),
    (torch.tensor([0.]), torch.tensor(-3.069)),
    (torch.tensor([1., 0., 0.]), torch.tensor(-8.901)),
])
def test_enum_log_prob_multiple_iarange(data, expected_log_prob):

    @poutine.enum(first_available_dim=1)
    @config_enumerate(default="parallel")
    @poutine.condition(data={"p": torch.tensor(0.4)})
    @poutine.broadcast
    def model(data):
        p = pyro.sample("p", dist.Uniform(0., 1.))
        y = pyro.sample("y", dist.Bernoulli(p))
        q = 0.5 + 0.25 * y
        with pyro.iarange("data1", len(data)):
            v = pyro.sample("v", dist.Bernoulli(q))
            pyro.sample("obs1", dist.Normal(2 * v, 1.), obs=data)
        with pyro.iarange("data2", len(data)):
            z = pyro.sample("z", dist.Bernoulli(q))
            pyro.sample("obs2", dist.Normal(2 * z - 1, 1.), obs=data)

    model_trace = poutine.trace(model).get_trace(data)
    print_debug_info(model_trace)
    trace_prob_evaluator = EnumTraceProbEvaluator(model_trace, True, 1)
    assert_equal(trace_prob_evaluator.log_prob(),
                 expected_log_prob,
                 prec=1e-3)


@pytest.mark.parametrize("data, expected_log_prob", [
    (torch.tensor([1.]), torch.tensor(-1.5478)),
    (torch.tensor([0.]), torch.tensor(-1.4189)),
    (torch.tensor([1., 0., 0.]), torch.tensor(-4.3857)),
])
def test_enum_log_prob_nested_iarange(data, expected_log_prob):

    @poutine.enum(first_available_dim=2)
    @config_enumerate(default="parallel")
    @poutine.condition(data={"p": torch.tensor(0.4)})
    @poutine.broadcast
    def model(data):
        p = pyro.sample("p", dist.Uniform(0., 1.))
        y = pyro.sample("y", dist.Bernoulli(p))
        q = 0.5 + 0.25 * y
        with pyro.iarange("intermediate", 1, dim=-2):
            v = pyro.sample("v", dist.Bernoulli(q))
            with pyro.iarange("data", len(data), dim=-1):
                r = 0.4 + 0.1 * v
                z = pyro.sample("z", dist.Bernoulli(r))
                pyro.sample("obs", dist.Normal(2 * z - 1, 1.), obs=data)

    model_trace = poutine.trace(model).get_trace(data)
    print_debug_info(model_trace)
    trace_prob_evaluator = EnumTraceProbEvaluator(model_trace, True, 2)
    assert_equal(trace_prob_evaluator.log_prob(),
                 expected_log_prob,
                 prec=1e-3)


def test_log_prob_eval_iterates_in_correct_order():
    @poutine.enum(first_available_dim=4)
    @config_enumerate(default="parallel")
    @poutine.condition(data={"p": torch.tensor(0.4)})
    @poutine.broadcast
    def model():
        outer = pyro.iarange("outer", 3, dim=-1)
        inner1 = pyro.iarange("inner1", 4, dim=-3)
        inner2 = pyro.iarange("inner2", 5, dim=-2)
        inner3 = pyro.iarange("inner3", 6, dim=-4)

        p = pyro.sample("p", dist.Uniform(0., 1.))
        y = pyro.sample("y", dist.Bernoulli(p))
        q = 0.5 + 0.25 * y
        with outer, inner2:
            z0 = pyro.sample("z0", dist.Bernoulli(q))
            pyro.sample("obs0", dist.Normal(2 * z0 - 1, 1.), obs=torch.ones(5, 3))
        with outer:
            v = pyro.sample("v", dist.Bernoulli(q))
            r = 0.4 + 0.1 * v
            with inner1, inner3:
                z1 = pyro.sample("z1", dist.Bernoulli(r))
                pyro.sample("obs1", dist.Normal(2 * z1 - 1, 1.), obs=torch.ones(6, 4, 1, 3))
            with inner2:
                z2 = pyro.sample("z2", dist.Bernoulli(r))
                pyro.sample("obs2", dist.Normal(2 * z2 - 1, 1.), obs=torch.ones(5, 3))

    model_trace = poutine.trace(model).get_trace()
    trace_prob_evaluator = EnumTraceProbEvaluator(model_trace, True, 4)
    trace_prob_evaluator.log_prob()
    assert_equal([stack.name for stack in reversed(trace_prob_evaluator._sorted_indep_stacks)],
                 ["inner3", "inner1", "inner2", "outer", "default"])


def test_model():
    @poutine.enum(first_available_dim=1)
    @config_enumerate(default="parallel")
    @poutine.condition(data={"p": torch.tensor(0.4)})
    def model():
        p = pyro.sample("p", dist.Beta(2., 2.))
        x = pyro.sample("x", dist.Bernoulli(p))
        with pyro.iarange("iarange_y", 2, dim=-1):
            pyro.sample("y", dist.Normal(0, 1), obs=x)
        with pyro.iarange("iarange_z", 3, dim=-2):
            pyro.sample("z", dist.Normal(0, 1), obs=x)

    model_trace = poutine.trace(model).get_trace()
    print_debug_info(model_trace)
    trace_prob_evaluator = EnumTraceProbEvaluator(model_trace, True, 1)
    print(trace_prob_evaluator.log_prob())
