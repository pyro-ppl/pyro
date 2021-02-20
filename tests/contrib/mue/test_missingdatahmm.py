# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.contrib.mue.missingdatahmm import MissingDataDiscreteHMM
from pyro.distributions import Categorical, DiscreteHMM


def test_hmm_log_prob():

    a0 = torch.tensor([0.9, 0.08, 0.02])
    a = torch.tensor([[0.1, 0.8, 0.1], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    e = torch.tensor([[0.99, 0.01], [0.01, 0.99], [0.5, 0.5]])

    x = torch.tensor([[0., 1.],
                      [1., 0.],
                      [0., 1.],
                      [0., 1.],
                      [1., 0.],
                      [0., 0.]])

    hmm_distr = MissingDataDiscreteHMM(torch.log(a0), torch.log(a),
                                       torch.log(e))
    lp = hmm_distr.log_prob(x)

    f = a0 * e[:, 1]
    f = torch.matmul(f, a) * e[:, 0]
    f = torch.matmul(f, a) * e[:, 1]
    f = torch.matmul(f, a) * e[:, 1]
    f = torch.matmul(f, a) * e[:, 0]
    chk_lp = torch.log(torch.sum(f))

    assert torch.allclose(lp, chk_lp)

    # Batch values.
    x = torch.cat([
        x[None, :, :],
        torch.tensor([[1., 0.],
                      [1., 0.],
                      [1., 0.],
                      [0., 0.],
                      [0., 0.],
                      [0., 0.]])[None, :, :]], dim=0)
    lp = hmm_distr.log_prob(x)

    f = a0 * e[:, 0]
    f = torch.matmul(f, a) * e[:, 0]
    f = torch.matmul(f, a) * e[:, 0]
    chk_lp = torch.cat([chk_lp[None], torch.log(torch.sum(f))[None]])

    assert torch.allclose(lp, chk_lp)

    # Batch both parameters and values.
    a0 = torch.cat([a0[None, :], torch.tensor([0.2, 0.7, 0.1])[None, :]])
    a = torch.cat([
        a[None, :, :],
        torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]]
                     )[None, :, :]], dim=0)
    e = torch.cat([
        e[None, :, :],
        torch.tensor([[0.4, 0.6], [0.99, 0.01], [0.7, 0.3]])[None, :, :]],
        dim=0)
    hmm_distr = MissingDataDiscreteHMM(torch.log(a0), torch.log(a),
                                       torch.log(e))
    lp = hmm_distr.log_prob(x)

    f = a0[1, :] * e[1, :, 0]
    f = torch.matmul(f, a[1, :, :]) * e[1, :, 0]
    f = torch.matmul(f, a[1, :, :]) * e[1, :, 0]
    chk_lp = torch.cat([chk_lp[0][None], torch.log(torch.sum(f))[None]])

    assert torch.allclose(lp, chk_lp)


@pytest.mark.parametrize('batch_initial', [False, True])
@pytest.mark.parametrize('batch_transition', [False, True])
@pytest.mark.parametrize('batch_observation', [False, True])
@pytest.mark.parametrize('batch_data', [False, True])
def test_shapes(batch_initial, batch_transition, batch_observation, batch_data):

    # Dimensions.
    batch_size = 3
    state_dim, observation_dim, num_steps = 4, 5, 6

    # Model initialization.
    initial_logits = torch.randn([batch_size]*batch_initial + [state_dim])
    initial_logits = (initial_logits -
                      initial_logits.logsumexp(-1, True))
    transition_logits = torch.randn([batch_size]*batch_transition
                                    + [state_dim, state_dim])
    transition_logits = (transition_logits -
                         transition_logits.logsumexp(-1, True))
    observation_logits = torch.randn([batch_size]*batch_observation
                                     + [state_dim, observation_dim])
    observation_logits = (observation_logits -
                          observation_logits.logsumexp(-1, True))

    hmm = MissingDataDiscreteHMM(initial_logits, transition_logits,
                                 observation_logits)

    # Random observations.
    value = (torch.randint(observation_dim,
                           [batch_size]*batch_data + [num_steps]).unsqueeze(-1)
             == torch.arange(observation_dim)).double()

    # Log probability.
    lp = hmm.log_prob(value)

    # Check shapes:
    if all([not batch_initial, not batch_transition, not batch_observation,
            not batch_data]):
        assert lp.shape == ()
    else:
        assert lp.shape == (batch_size,)


@pytest.mark.parametrize('batch_initial', [False, True])
@pytest.mark.parametrize('batch_transition', [False, True])
@pytest.mark.parametrize('batch_observation', [False, True])
@pytest.mark.parametrize('batch_data', [False, True])
def test_DiscreteHMM_comparison(batch_initial, batch_transition,
                                batch_observation, batch_data):
    # Dimensions.
    batch_size = 3
    state_dim, observation_dim, num_steps = 4, 5, 6

    # -- Model setup --.
    transition_logits_vldhmm = torch.randn([batch_size]*batch_transition
                                           + [state_dim, state_dim])
    transition_logits_vldhmm = (transition_logits_vldhmm -
                                transition_logits_vldhmm.logsumexp(-1, True))
    # Adjust for DiscreteHMM broadcasting convention.
    transition_logits_dhmm = transition_logits_vldhmm.unsqueeze(-3)
    # Convert between discrete HMM convention for initial state and variable
    # length HMM convention.
    initial_logits_dhmm = torch.randn([batch_size]*batch_initial + [state_dim])
    initial_logits_dhmm = (initial_logits_dhmm -
                           initial_logits_dhmm.logsumexp(-1, True))
    initial_logits_vldhmm = (initial_logits_dhmm.unsqueeze(-1) +
                             transition_logits_vldhmm).logsumexp(-2)
    observation_logits = torch.randn([batch_size]*batch_observation
                                     + [state_dim, observation_dim])
    observation_logits = (observation_logits -
                          observation_logits.logsumexp(-1, True))
    # Create distribution object for DiscreteHMM
    observation_dist = Categorical(logits=observation_logits.unsqueeze(-3))

    vldhmm = MissingDataDiscreteHMM(initial_logits_vldhmm,
                                    transition_logits_vldhmm,
                                    observation_logits)
    dhmm = DiscreteHMM(initial_logits_dhmm, transition_logits_dhmm,
                       observation_dist)

    # Random observations.
    value = torch.randint(observation_dim,
                          [batch_size]*batch_data + [num_steps])
    value_oh = (value.unsqueeze(-1)
                == torch.arange(observation_dim)).double()

    # -- Check. --
    # Log probability.
    lp_vldhmm = vldhmm.log_prob(value_oh)
    lp_dhmm = dhmm.log_prob(value)
    # Shapes.
    if all([not batch_initial, not batch_transition, not batch_observation,
            not batch_data]):
        assert lp_vldhmm.shape == ()
    else:
        assert lp_vldhmm.shape == (batch_size,)
    # Values.
    assert torch.allclose(lp_vldhmm, lp_dhmm)
