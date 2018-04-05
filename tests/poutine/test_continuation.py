from __future__ import absolute_import, division, print_function

import logging
import math

import pytest
import torch
from torch.distributions import constraints, kl_divergence

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist
from pyro.infer import config_enumerate
from tests.common import assert_equal

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("width,num_steps,markov_order", [
    (2, 5, 1),
])
def test_chain_no_model_iarange(width, num_steps, markov_order):

    def markov_model(width, num_steps, markov_order):
        transition_probs = []
        for k in range(markov_order):
            transition_probs.append(
                pyro.param("p{}".format(k), torch.ones(width, width) / float(width)))
        init_prob = torch.ones(width) / float(width)

        samples = []
        for i in range(num_steps):
            if i == 0:
                prob = init_prob
            else:
                terms = [transition_probs[j][z]
                         for j, z in enumerate(samples[-min(markov_order, i):])]
                if i == num_steps-1:
                    print("sample shapes", [sample.shape for sample in samples])
                    print("term shapes", [term.shape for term in terms])
                prob = sum(reversed(terms))
            samples.append(pyro.sample("z{}".format(i), dist.Categorical(prob)))
        return samples

    assert len(markov_model(width, num_steps, markov_order)) == num_steps

    tp1 = poutine.trace(markov_model)

    tr1 = tp1.get_trace(width, num_steps, markov_order)

    assert "z{}".format(num_steps-1) in tr1

    tp = poutine.trace(
        poutine.enum(
            config_enumerate(default="parallel")(markov_model),
            first_available_dim=1))

    tr = tp.get_trace(width, num_steps, markov_order)

    counter = 0
    for name, site in tr.nodes.items():
        if site["type"] == "sample":
            if site["infer"].get("enumerate") == "parallel":
                counter = min(counter + 1, markov_order + 1)
                assert_equal(len(site["cond_indep_stack"]), counter)
                for frame in site["cond_indep_stack"]:
                    assert_equal(site["value"].shape[frame.dim], width)
