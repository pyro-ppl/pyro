# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import io
import logging

import pytest
import torch

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import config_enumerate
from pyro.infer.mcmc import HMC, NUTS
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc.util import TraceEinsumEvaluator, TraceTreeEvaluator, initialize_model
from pyro.infer.reparam import LatentStableReparam
from pyro.poutine.subsample_messenger import _Subsample
from tests.common import assert_close, assert_equal, xfail_param

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

    def gmm():
        data = torch.tensor([0., 0., 3., 3., 3., 5., 5.])
        mix_proportions = pyro.sample("phi", dist.Dirichlet(torch.ones(3)))
        cluster_means = pyro.sample("cluster_means", dist.Normal(torch.arange(3.), 1.))
        with pyro.plate("data", data.shape[0]):
            assignments = pyro.sample("assignments", dist.Categorical(mix_proportions))
            pyro.sample("obs", dist.Normal(cluster_means[assignments], 1.), obs=data)
        return cluster_means

    mcmc_kernel = kernel(gmm, **kwargs)
    # Error due to non finite value for `max_plate_nesting`.
    assert_error(mcmc_kernel)
    # Error due to batch dims not inside plate.
    mcmc_kernel = kernel(gmm, max_plate_nesting=1, **kwargs)
    assert_error(mcmc_kernel)


@pytest.mark.parametrize("kernel, kwargs", [
    (HMC, {"adapt_step_size": True, "num_steps": 3}),
    (NUTS, {"adapt_step_size": True}),
])
def test_model_error_enum_dim_clash(kernel, kwargs):

    def gmm():
        data = torch.tensor([0., 0., 3., 3., 3., 5., 5.])
        with pyro.plate("num_clusters", 3):
            mix_proportions = pyro.sample("phi", dist.Dirichlet(torch.tensor(1.)))
            cluster_means = pyro.sample("cluster_means", dist.Normal(torch.arange(3.), 1.))
        with pyro.plate("data", data.shape[0]):
            assignments = pyro.sample("assignments", dist.Categorical(mix_proportions))
            pyro.sample("obs", dist.Normal(cluster_means[assignments], 1.), obs=data)
        return cluster_means

    mcmc_kernel = kernel(gmm, max_plate_nesting=0, **kwargs)
    assert_error(mcmc_kernel)


def test_log_prob_eval_iterates_in_correct_order():
    @poutine.enum(first_available_dim=-5)
    @config_enumerate
    @poutine.condition(data={"p": torch.tensor(0.4)})
    def model():
        outer = pyro.plate("outer", 3, dim=-1)
        inner1 = pyro.plate("inner1", 4, dim=-3)
        inner2 = pyro.plate("inner2", 5, dim=-2)
        inner3 = pyro.plate("inner3", 6, dim=-4)

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
    trace_prob_evaluator = TraceTreeEvaluator(model_trace, True, 4)
    trace_prob_evaluator.log_prob(model_trace)
    plate_dims, enum_dims = [], []
    for key in reversed(sorted(trace_prob_evaluator._log_probs.keys(), key=lambda x: (len(x), x))):
        plate_dims.append(trace_prob_evaluator._plate_dims[key])
        enum_dims.append(trace_prob_evaluator._enum_dims[key])
    # The reduction operation returns a singleton with dimensions preserved.
    assert not any(i != 1 for i in trace_prob_evaluator._aggregate_log_probs(frozenset()).shape)
    assert plate_dims == [[-4, -3], [-2], [-1], []]
    assert enum_dims, [[-8], [-9, -6], [-7], [-5]]


@pytest.mark.parametrize("Eval", [TraceTreeEvaluator, TraceEinsumEvaluator])
def test_all_discrete_sites_log_prob(Eval):
    p = 0.3

    @poutine.enum(first_available_dim=-4)
    @config_enumerate
    def model():
        d = dist.Bernoulli(p)
        context1 = pyro.plate("outer", 2, dim=-1)
        context2 = pyro.plate("inner1", 1, dim=-2)
        context3 = pyro.plate("inner2", 1, dim=-3)
        pyro.sample("w", d)
        with context1:
            pyro.sample("x", d)
        with context1, context2:
            pyro.sample("y", d)
        with context1, context3:
            pyro.sample("z", d)

    model_trace = poutine.trace(model).get_trace()
    print_debug_info(model_trace)
    trace_prob_evaluator = Eval(model_trace, True, 3)
    # all discrete sites enumerated out.
    assert_equal(trace_prob_evaluator.log_prob(model_trace), torch.tensor(0.))


@pytest.mark.parametrize("Eval", [TraceTreeEvaluator,
                                  xfail_param(TraceEinsumEvaluator, reason="TODO: Debug this failure case.")])
def test_enumeration_in_tree(Eval):
    @poutine.enum(first_available_dim=-5)
    @config_enumerate
    @poutine.condition(data={"sample1": torch.tensor(0.),
                             "sample2": torch.tensor(1.),
                             "sample3": torch.tensor(2.)})
    def model():
        outer = pyro.plate("outer", 2, dim=-1)
        inner1 = pyro.plate("inner1", 2, dim=-3)
        inner2 = pyro.plate("inner2", 3, dim=-2)
        inner3 = pyro.plate("inner3", 2, dim=-4)

        d = dist.Bernoulli(0.3)
        n = dist.Normal(0., 1.)
        pyro.sample("y", d)
        pyro.sample("sample1", n)
        with outer, inner2:
            pyro.sample("z0", d)
            pyro.sample("sample2", n)
        with outer:
            pyro.sample("z1", d)
            pyro.sample("sample3", n)
            with inner1, inner3:
                pyro.sample("z2", d)
            with inner2:
                pyro.sample("z3", d)

    model_trace = poutine.trace(model).get_trace()
    print_debug_info(model_trace)
    trace_prob_evaluator = Eval(model_trace, True, 4)
    # p_n(0.) * p_n(2.)^2 * p_n(1.)^6
    assert_equal(trace_prob_evaluator.log_prob(model_trace), torch.tensor(-15.2704), prec=1e-4)


@pytest.mark.xfail(reason="Enumeration currently does not work for general DAGs")
@pytest.mark.parametrize("Eval", [TraceTreeEvaluator, TraceEinsumEvaluator])
def test_enumeration_in_dag(Eval):
    p = 0.3

    @poutine.enum(first_available_dim=-3)
    @config_enumerate
    @poutine.condition(data={"b": torch.tensor(0.4), "c": torch.tensor(0.4)})
    def model():
        d = dist.Bernoulli(p)
        context1 = pyro.plate("outer", 3, dim=-1)
        context2 = pyro.plate("inner", 2, dim=-2)
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
    trace_prob_evaluator = Eval(model_trace, True, 2)
    assert_equal(trace_prob_evaluator.log_prob(model_trace), torch.tensor(0.16196))  # p_beta(0.3)^3


@pytest.mark.parametrize("data, expected_log_prob", [
    (torch.tensor([1.]), torch.tensor(-1.3434)),
    (torch.tensor([0.]), torch.tensor(-1.4189)),
    (torch.tensor([1., 0., 0.]), torch.tensor(-4.1813)),
])
@pytest.mark.parametrize("Eval", [TraceTreeEvaluator, TraceEinsumEvaluator])
def test_enum_log_prob_continuous_observed(data, expected_log_prob, Eval):

    @poutine.enum(first_available_dim=-2)
    @config_enumerate
    @poutine.condition(data={"p": torch.tensor(0.4)})
    def model(data):
        p = pyro.sample("p", dist.Uniform(0., 1.))
        y = pyro.sample("y", dist.Bernoulli(p))
        q = 0.5 + 0.25 * y
        with pyro.plate("data", len(data)):
            z = pyro.sample("z", dist.Bernoulli(q))
            mean = 2 * z - 1
            pyro.sample("obs", dist.Normal(mean, 1.), obs=data)

    model_trace = poutine.trace(model).get_trace(data)
    print_debug_info(model_trace)
    trace_prob_evaluator = Eval(model_trace, True, 1)
    assert_equal(trace_prob_evaluator.log_prob(model_trace),
                 expected_log_prob,
                 prec=1e-3)


@pytest.mark.parametrize("data, expected_log_prob", [
    (torch.tensor([1.]), torch.tensor(-3.5237)),
    (torch.tensor([0.]), torch.tensor(-3.7091)),
    (torch.tensor([1., 1.]), torch.tensor(-3.9699)),
    (torch.tensor([1., 0., 0.]), torch.tensor(-5.3357)),
])
@pytest.mark.parametrize("Eval", [TraceTreeEvaluator, TraceEinsumEvaluator])
def test_enum_log_prob_continuous_sampled(data, expected_log_prob, Eval):

    @poutine.enum(first_available_dim=-2)
    @config_enumerate
    @poutine.condition(data={"p": torch.tensor(0.4),
                             "n": torch.tensor([[1.], [-1.]])})
    def model(data):
        p = pyro.sample("p", dist.Uniform(0., 1.))
        y = pyro.sample("y", dist.Bernoulli(p))
        mean = 2 * y - 1
        n = pyro.sample("n", dist.Normal(mean, 1.))
        with pyro.plate("data", len(data)):
            pyro.sample("obs", dist.Bernoulli(torch.sigmoid(n)), obs=data)

    model_trace = poutine.trace(model).get_trace(data)
    print_debug_info(model_trace)
    trace_prob_evaluator = Eval(model_trace, True, 1)
    assert_equal(trace_prob_evaluator.log_prob(model_trace),
                 expected_log_prob,
                 prec=1e-3)


@pytest.mark.parametrize("data, expected_log_prob", [
    (torch.tensor([1.]), torch.tensor(-0.5108)),
    (torch.tensor([1., 1.]), torch.tensor(-0.9808)),
    (torch.tensor([1., 0., 0.]), torch.tensor(-2.3671)),
])
@pytest.mark.parametrize("Eval", [TraceTreeEvaluator, TraceEinsumEvaluator])
def test_enum_log_prob_discrete_observed(data, expected_log_prob, Eval):

    @poutine.enum(first_available_dim=-2)
    @config_enumerate
    @poutine.condition(data={"p": torch.tensor(0.4)})
    def model(data):
        p = pyro.sample("p", dist.Uniform(0., 1.))
        y = pyro.sample("y", dist.Bernoulli(p))
        q = 0.25 * y + 0.5
        with pyro.plate("data", len(data)):
            pyro.sample("obs", dist.Bernoulli(q), obs=data)

    model_trace = poutine.trace(model).get_trace(data)
    print_debug_info(model_trace)
    trace_prob_evaluator = Eval(model_trace, True, 1)
    assert_equal(trace_prob_evaluator.log_prob(model_trace),
                 expected_log_prob,
                 prec=1e-3)


@pytest.mark.parametrize("data, expected_log_prob", [
    (torch.tensor([1.]), torch.tensor(-1.15)),
    (torch.tensor([0.]), torch.tensor(-1.46)),
    (torch.tensor([1., 1.]), torch.tensor(-2.1998)),
])
@pytest.mark.parametrize("Eval", [TraceTreeEvaluator, TraceEinsumEvaluator])
def test_enum_log_prob_multiple_plate(data, expected_log_prob, Eval):

    @poutine.enum(first_available_dim=-2)
    @config_enumerate
    @poutine.condition(data={"p": torch.tensor(0.4)})
    def model(data):
        p = pyro.sample("p", dist.Beta(1.1, 1.1))
        y = pyro.sample("y", dist.Bernoulli(p))
        q = 0.5 + 0.25 * y
        r = 0.4 + 0.2 * y
        with pyro.plate("data1", len(data)):
            pyro.sample("obs1", dist.Bernoulli(q), obs=data)
        with pyro.plate("data2", len(data)):
            pyro.sample("obs2", dist.Bernoulli(r), obs=data)

    model_trace = poutine.trace(model).get_trace(data)
    print_debug_info(model_trace)
    trace_prob_evaluator = Eval(model_trace, True, 1)
    assert_equal(trace_prob_evaluator.log_prob(model_trace),
                 expected_log_prob,
                 prec=1e-3)


@pytest.mark.parametrize("data, expected_log_prob", [
    (torch.tensor([1.]), torch.tensor(-1.5478)),
    (torch.tensor([0.]), torch.tensor(-1.4189)),
    (torch.tensor([1., 0., 0.]), torch.tensor(-4.3857)),
])
@pytest.mark.parametrize("Eval", [TraceTreeEvaluator, TraceEinsumEvaluator])
def test_enum_log_prob_nested_plate(data, expected_log_prob, Eval):

    @poutine.enum(first_available_dim=-3)
    @config_enumerate
    @poutine.condition(data={"p": torch.tensor(0.4)})
    def model(data):
        p = pyro.sample("p", dist.Uniform(0., 1.))
        y = pyro.sample("y", dist.Bernoulli(p))
        q = 0.5 + 0.25 * y
        with pyro.plate("intermediate", 1, dim=-2):
            v = pyro.sample("v", dist.Bernoulli(q))
            with pyro.plate("data", len(data), dim=-1):
                r = 0.4 + 0.1 * v
                z = pyro.sample("z", dist.Bernoulli(r))
                pyro.sample("obs", dist.Normal(2 * z - 1, 1.), obs=data)

    model_trace = poutine.trace(model).get_trace(data)
    print_debug_info(model_trace)
    trace_prob_evaluator = Eval(model_trace, True, 2)
    assert_equal(trace_prob_evaluator.log_prob(model_trace),
                 expected_log_prob,
                 prec=1e-3)


def _beta_bernoulli(data):
    alpha = torch.tensor([1.1, 1.1])
    beta = torch.tensor([1.1, 1.1])
    p_latent = pyro.sample('p_latent', dist.Beta(alpha, beta))
    with pyro.plate('data', data.shape[0], dim=-2):
        pyro.sample('obs', dist.Bernoulli(p_latent), obs=data)
    return p_latent


@pytest.mark.parametrize('jit', [False, True])
def test_potential_fn_pickling(jit):
    data = dist.Bernoulli(torch.tensor([0.8, 0.2])).sample(sample_shape=(torch.Size((1000,))))
    _, potential_fn, _, _ = initialize_model(_beta_bernoulli, (data,), jit_compile=jit,
                                             skip_jit_warnings=True)
    test_data = {'p_latent': torch.tensor([0.2, 0.6])}
    buffer = io.BytesIO()
    torch.save(potential_fn, buffer)
    buffer.seek(0)
    deser_potential_fn = torch.load(buffer)
    assert_close(deser_potential_fn(test_data), potential_fn(test_data))


@pytest.mark.parametrize("kernel, kwargs", [
    (HMC, {"adapt_step_size": True, "num_steps": 3}),
    (NUTS, {"adapt_step_size": True}),
])
def test_reparam_stable(kernel, kwargs):

    @poutine.reparam(config={"z": LatentStableReparam()})
    def model():
        stability = pyro.sample("stability", dist.Uniform(0., 2.))
        skew = pyro.sample("skew", dist.Uniform(-1., 1.))
        y = pyro.sample("z", dist.Stable(stability, skew))
        pyro.sample("x", dist.Poisson(y.abs()), obs=torch.tensor(1.))

    mcmc_kernel = kernel(model, max_plate_nesting=0, **kwargs)
    assert_ok(mcmc_kernel)
