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

    x = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    )

    hmm_distr = MissingDataDiscreteHMM(torch.log(a0), torch.log(a), torch.log(e))
    lp = hmm_distr.log_prob(x)

    f = a0 * e[:, 1]
    f = torch.matmul(f, a) * e[:, 0]
    f = torch.matmul(f, a) * e[:, 1]
    f = torch.matmul(f, a) * e[:, 1]
    f = torch.matmul(f, a) * e[:, 0]
    expected_lp = torch.log(torch.sum(f))

    assert torch.allclose(lp, expected_lp)

    # Batch values.
    x = torch.cat(
        [
            x[None, :, :],
            torch.tensor(
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            )[None, :, :],
        ],
        dim=0,
    )
    lp = hmm_distr.log_prob(x)

    f = a0 * e[:, 0]
    f = torch.matmul(f, a) * e[:, 0]
    f = torch.matmul(f, a) * e[:, 0]
    expected_lp = torch.cat([expected_lp[None], torch.log(torch.sum(f))[None]])

    assert torch.allclose(lp, expected_lp)

    # Batch both parameters and values.
    a0 = torch.cat([a0[None, :], torch.tensor([0.2, 0.7, 0.1])[None, :]])
    a = torch.cat(
        [
            a[None, :, :],
            torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])[
                None, :, :
            ],
        ],
        dim=0,
    )
    e = torch.cat(
        [
            e[None, :, :],
            torch.tensor([[0.4, 0.6], [0.99, 0.01], [0.7, 0.3]])[None, :, :],
        ],
        dim=0,
    )
    hmm_distr = MissingDataDiscreteHMM(torch.log(a0), torch.log(a), torch.log(e))
    lp = hmm_distr.log_prob(x)

    f = a0[1, :] * e[1, :, 0]
    f = torch.matmul(f, a[1, :, :]) * e[1, :, 0]
    f = torch.matmul(f, a[1, :, :]) * e[1, :, 0]
    expected_lp = torch.cat([expected_lp[0][None], torch.log(torch.sum(f))[None]])

    assert torch.allclose(lp, expected_lp)


@pytest.mark.parametrize("batch_initial", [False, True])
@pytest.mark.parametrize("batch_transition", [False, True])
@pytest.mark.parametrize("batch_observation", [False, True])
@pytest.mark.parametrize("batch_data", [False, True])
def test_shapes(batch_initial, batch_transition, batch_observation, batch_data):
    # Dimensions.
    batch_size = 3
    state_dim, observation_dim, num_steps = 4, 5, 6

    # Model initialization.
    initial_logits = torch.randn([batch_size] * batch_initial + [state_dim])
    initial_logits = initial_logits - initial_logits.logsumexp(-1, True)
    transition_logits = torch.randn(
        [batch_size] * batch_transition + [state_dim, state_dim]
    )
    transition_logits = transition_logits - transition_logits.logsumexp(-1, True)
    observation_logits = torch.randn(
        [batch_size] * batch_observation + [state_dim, observation_dim]
    )
    observation_logits = observation_logits - observation_logits.logsumexp(-1, True)

    hmm = MissingDataDiscreteHMM(initial_logits, transition_logits, observation_logits)

    # Random observations.
    value = (
        torch.randint(
            observation_dim, [batch_size] * batch_data + [num_steps]
        ).unsqueeze(-1)
        == torch.arange(observation_dim)
    ).double()

    # Log probability.
    lp = hmm.log_prob(value)

    # Check shapes:
    if all(
        [not batch_initial, not batch_transition, not batch_observation, not batch_data]
    ):
        assert lp.shape == ()
    else:
        assert lp.shape == (batch_size,)


@pytest.mark.parametrize("batch_initial", [False, True])
@pytest.mark.parametrize("batch_transition", [False, True])
@pytest.mark.parametrize("batch_observation", [False, True])
@pytest.mark.parametrize("batch_data", [False, True])
def test_DiscreteHMM_comparison(
    batch_initial, batch_transition, batch_observation, batch_data
):
    # Dimensions.
    batch_size = 3
    state_dim, observation_dim, num_steps = 4, 5, 6

    # -- Model setup --.
    transition_logits_vldhmm = torch.randn(
        [batch_size] * batch_transition + [state_dim, state_dim]
    )
    transition_logits_vldhmm = (
        transition_logits_vldhmm - transition_logits_vldhmm.logsumexp(-1, True)
    )
    # Adjust for DiscreteHMM broadcasting convention.
    transition_logits_dhmm = transition_logits_vldhmm.unsqueeze(-3)
    # Convert between discrete HMM convention for initial state and variable
    # length HMM convention.
    initial_logits_dhmm = torch.randn([batch_size] * batch_initial + [state_dim])
    initial_logits_dhmm = initial_logits_dhmm - initial_logits_dhmm.logsumexp(-1, True)
    initial_logits_vldhmm = (
        initial_logits_dhmm.unsqueeze(-1) + transition_logits_vldhmm
    ).logsumexp(-2)
    observation_logits = torch.randn(
        [batch_size] * batch_observation + [state_dim, observation_dim]
    )
    observation_logits = observation_logits - observation_logits.logsumexp(-1, True)
    # Create distribution object for DiscreteHMM
    observation_dist = Categorical(logits=observation_logits.unsqueeze(-3))

    vldhmm = MissingDataDiscreteHMM(
        initial_logits_vldhmm, transition_logits_vldhmm, observation_logits
    )
    dhmm = DiscreteHMM(initial_logits_dhmm, transition_logits_dhmm, observation_dist)

    # Random observations.
    value = torch.randint(observation_dim, [batch_size] * batch_data + [num_steps])
    value_oh = (value.unsqueeze(-1) == torch.arange(observation_dim)).double()

    # -- Check. --
    # Log probability.
    lp_vldhmm = vldhmm.log_prob(value_oh)
    lp_dhmm = dhmm.log_prob(value)
    # Shapes.
    if all(
        [not batch_initial, not batch_transition, not batch_observation, not batch_data]
    ):
        assert lp_vldhmm.shape == ()
    else:
        assert lp_vldhmm.shape == (batch_size,)
    # Values.
    assert torch.allclose(lp_vldhmm, lp_dhmm)
    # Filter.
    filter_dhmm = dhmm.filter(value)
    filter_vldhmm = vldhmm.filter(value_oh)
    assert torch.allclose(filter_dhmm.logits, filter_vldhmm[..., -1, :])
    # Check other computations run.
    vldhmm.sample(value_oh.shape[:-1])
    vldhmm.smooth(value_oh)
    vldhmm.sample_states(value_oh)
    map_states = vldhmm.map_states(value_oh)
    print(value_oh.shape, map_states.shape)
    vldhmm.sample_given_states(map_states)


@pytest.mark.parametrize("batch_data", [False, True])
def test_samples(batch_data):
    initial_logits = torch.tensor([-100, 0, -100, -100], dtype=torch.float64)
    transition_logits = torch.tensor(
        [
            [-100, -100, 0, -100],
            [-100, -100, -100, 0],
            [0, -100, -100, -100],
            [-100, 0, -100, -100],
        ],
        dtype=torch.float64,
    )
    obs_logits = torch.tensor(
        [[0, -100, -100], [-100, 0, -100], [-100, -100, 0], [-100, -100, 0]],
        dtype=torch.float64,
    )
    if batch_data:
        initial_logits = torch.tensor(
            [[-100, 0, -100, -100], [0, -100, -100, -100]], dtype=torch.float64
        )
        transition_logits = transition_logits * torch.ones(
            [2] + list(transition_logits.shape)
        )
        obs_logits = obs_logits * torch.ones([2] + list(obs_logits.shape))

    model = MissingDataDiscreteHMM(initial_logits, transition_logits, obs_logits)

    if not batch_data:
        sample = model.sample(torch.Size([3]))
        print(sample)
        assert torch.allclose(
            sample, torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        )
    else:
        sample = model.sample(torch.Size([2, 3]))
        print(sample[0, :, :])
        assert torch.allclose(
            sample[0, :, :],
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        )
        print(sample[1, :, :])
        assert torch.allclose(
            sample[1, :, :],
            torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        )


def indiv_filter(a0, a, e, x):
    alph = torch.zeros((x.shape[0], a0.shape[0]))
    for j in range(a0.shape[0]):
        vec = a0[j]
        if torch.sum(x[0, :]) > 0.5:
            vec = vec * torch.dot(x[0, :], e[j, :])
        alph[0, j] = vec
    alph[0, :] = alph[0, :] / torch.sum(alph[0, :])
    for t in range(1, x.shape[0]):
        for j in range(a0.shape[0]):
            vec = torch.sum(alph[t - 1, :] * a[:, j])
            if torch.sum(x[t, :]) > 0.5:
                vec = vec * torch.dot(x[t, :], e[j, :])
            alph[t, j] = vec
        alph[t, :] = alph[t, :] / torch.sum(alph[t, :])
    return torch.log(alph)


def indiv_smooth(a0, a, e, x):
    alph = indiv_filter(a0, a, e, x)
    beta = torch.zeros(alph.shape)
    beta[-1, :] = 1.0
    for t in range(alph.shape[0] - 1, 0, -1):
        for i in range(a0.shape[0]):
            for j in range(a0.shape[0]):
                vec = beta[t, j] * a[i, j]
                if torch.sum(x[t, :]) > 0.5:
                    vec = vec * torch.dot(x[t, :], e[j, :])
                beta[t - 1, i] += vec
    smooth = torch.exp(alph) * beta
    smooth = smooth / torch.sum(smooth, -1, True)
    return torch.log(smooth)


def indiv_map_states(a0, a, e, x):
    # Viterbi algorithm, implemented without batching or vector operations.

    delta = torch.zeros((x.shape[0], a0.shape[0]))
    for j in range(a0.shape[0]):
        vec = a0[j]
        if torch.sum(x[0, :]) > 0.5:
            vec = vec * torch.dot(x[0, :], e[j, :])
        delta[0, j] = vec
    traceback = torch.zeros((x.shape[0], a0.shape[0]), dtype=torch.long)
    for t in range(1, x.shape[0]):
        for j in range(a0.shape[0]):
            vec = delta[t - 1, :] * a[:, j]
            if torch.sum(x[t, :]) > 0.5:
                vec = vec * torch.dot(x[t, :], e[j, :])
            delta[t, j] = torch.max(vec)
            traceback[t, j] = torch.argmax(vec)
    expected_map_states = torch.zeros(x.shape[0], dtype=torch.long)
    expected_map_states[-1] = torch.argmax(delta[-1, :])
    for t in range(x.shape[0] - 1, 0, -1):
        expected_map_states[t - 1] = traceback[t, expected_map_states[t]]

    return expected_map_states


def test_state_infer():
    # HMM parameters.
    a0 = torch.tensor([0.9, 0.08, 0.02])
    a = torch.tensor([[0.1, 0.8, 0.1], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    e = torch.tensor([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
    # Observed value.
    x = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    )

    expected_map_states = indiv_map_states(a0, a, e, x)
    expected_filter = indiv_filter(a0, a, e, x)
    expected_smooth = indiv_smooth(a0, a, e, x)

    hmm_distr = MissingDataDiscreteHMM(torch.log(a0), torch.log(a), torch.log(e))
    map_states = hmm_distr.map_states(x)
    filter = hmm_distr.filter(x)
    smooth = hmm_distr.smooth(x)

    assert torch.allclose(map_states, expected_map_states)
    assert torch.allclose(filter, expected_filter)
    assert torch.allclose(smooth, expected_smooth)

    # Batch values.
    x = torch.cat(
        [
            x[None, :, :],
            torch.tensor(
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
            )[None, :, :],
        ],
        dim=0,
    )
    map_states = hmm_distr.map_states(x)
    filter = hmm_distr.filter(x)
    smooth = hmm_distr.smooth(x)

    expected_map_states = torch.cat(
        [
            indiv_map_states(a0, a, e, x[0])[None, :],
            indiv_map_states(a0, a, e, x[1])[None, :],
        ],
        -2,
    )
    expected_filter = torch.cat(
        [
            indiv_filter(a0, a, e, x[0])[None, :, :],
            indiv_filter(a0, a, e, x[1])[None, :, :],
        ],
        -3,
    )
    expected_smooth = torch.cat(
        [
            indiv_smooth(a0, a, e, x[0])[None, :, :],
            indiv_smooth(a0, a, e, x[1])[None, :, :],
        ],
        -3,
    )

    assert torch.allclose(map_states, expected_map_states)
    assert torch.allclose(filter, expected_filter)
    assert torch.allclose(smooth, expected_smooth)

    # Batch parameters.
    a0 = torch.cat([a0[None, :], torch.tensor([0.2, 0.7, 0.1])[None, :]])
    a = torch.cat(
        [
            a[None, :, :],
            torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8]])[
                None, :, :
            ],
        ],
        dim=0,
    )
    e = torch.cat(
        [
            e[None, :, :],
            torch.tensor([[0.4, 0.6], [0.99, 0.01], [0.7, 0.3]])[None, :, :],
        ],
        dim=0,
    )
    hmm_distr = MissingDataDiscreteHMM(torch.log(a0), torch.log(a), torch.log(e))
    map_states = hmm_distr.map_states(x[1])
    filter = hmm_distr.filter(x[1])
    smooth = hmm_distr.smooth(x[1])

    expected_map_states = torch.cat(
        [
            indiv_map_states(a0[0], a[0], e[0], x[1])[None, :],
            indiv_map_states(a0[1], a[1], e[1], x[1])[None, :],
        ],
        -2,
    )
    expected_filter = torch.cat(
        [
            indiv_filter(a0[0], a[0], e[0], x[1])[None, :, :],
            indiv_filter(a0[1], a[1], e[1], x[1])[None, :, :],
        ],
        -3,
    )
    expected_smooth = torch.cat(
        [
            indiv_smooth(a0[0], a[0], e[0], x[1])[None, :, :],
            indiv_smooth(a0[1], a[1], e[1], x[1])[None, :, :],
        ],
        -3,
    )

    assert torch.allclose(map_states, expected_map_states)
    assert torch.allclose(filter, expected_filter)
    assert torch.allclose(smooth, expected_smooth)

    # Batch both parameters and values.
    map_states = hmm_distr.map_states(x)
    filter = hmm_distr.filter(x)
    smooth = hmm_distr.smooth(x)

    expected_map_states = torch.cat(
        [
            indiv_map_states(a0[0], a[0], e[0], x[0])[None, :],
            indiv_map_states(a0[1], a[1], e[1], x[1])[None, :],
        ],
        -2,
    )
    expected_filter = torch.cat(
        [
            indiv_filter(a0[0], a[0], e[0], x[0])[None, :, :],
            indiv_filter(a0[1], a[1], e[1], x[1])[None, :, :],
        ],
        -3,
    )
    expected_smooth = torch.cat(
        [
            indiv_smooth(a0[0], a[0], e[0], x[0])[None, :, :],
            indiv_smooth(a0[1], a[1], e[1], x[1])[None, :, :],
        ],
        -3,
    )

    assert torch.allclose(map_states, expected_map_states)
    assert torch.allclose(filter, expected_filter)
    assert torch.allclose(smooth, expected_smooth)


def test_sample_given_states():
    a0 = torch.tensor([0.9, 0.08, 0.02])
    a = torch.tensor([[0.1, 0.8, 0.1], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    eps = 1e-10
    # Effectively deterministic to check sampler.
    e = torch.tensor([[1 - eps, eps], [eps, 1 - eps], [eps, 1 - eps]])

    map_states = torch.tensor([0, 2, 1, 0], dtype=torch.long)
    hmm_distr = MissingDataDiscreteHMM(torch.log(a0), torch.log(a), torch.log(e))
    sample = hmm_distr.sample_given_states(map_states)
    expected_sample = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    assert torch.allclose(sample, expected_sample)

    # Batch values
    map_states = torch.tensor([[0, 2, 1, 0], [0, 0, 0, 1]], dtype=torch.long)
    sample = hmm_distr.sample_given_states(map_states)
    expected_sample = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ]
    )
    assert torch.allclose(sample, expected_sample)

    # Batch parameters
    e = torch.cat(
        [
            e[None, :, :],
            torch.tensor([[eps, 1 - eps], [eps, 1 - eps], [1 - eps, eps]])[None, :, :],
        ],
        dim=0,
    )
    hmm_distr = MissingDataDiscreteHMM(torch.log(a0), torch.log(a), torch.log(e))
    sample = hmm_distr.sample_given_states(map_states[0])
    expected_sample = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )
    assert torch.allclose(sample, expected_sample)

    # Batch parameters and values.
    sample = hmm_distr.sample_given_states(map_states)
    expected_sample = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ]
    )
    assert torch.allclose(sample, expected_sample)


def test_sample_states():
    # Effectively deterministic to check sampler.
    eps = 1e-10
    a0 = torch.tensor([1 - eps, eps / 2, eps / 2])
    a = torch.tensor(
        [
            [eps / 2, 1 - eps, eps / 2],
            [eps, 0.5 - eps / 2, 0.5 - eps / 2],
            [eps, 0.5 - eps / 2, 0.5 - eps / 2],
        ]
    )
    e = torch.tensor([[1 - eps, eps], [1 - eps, eps], [eps, 1 - eps]])
    x = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    hmm_distr = MissingDataDiscreteHMM(torch.log(a0), torch.log(a), torch.log(e))
    states = hmm_distr.sample_states(x)
    expected_states = torch.tensor([0, 1, 2, 2])
    assert torch.allclose(states, expected_states)

    # Batch values.
    x = torch.cat(
        [
            x[None, :, :],
            torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])[None, :, :],
        ],
        dim=0,
    )
    states = hmm_distr.sample_states(x)
    expected_states = torch.tensor([[0, 1, 2, 2], [0, 1, 2, 1]])
    assert torch.allclose(states, expected_states)

    # Batch parameters
    a0 = torch.cat([a0[None, :], torch.tensor([eps / 2, 1 - eps, eps / 2])[None, :]])
    a = torch.cat(
        [
            a[None, :, :],
            torch.tensor(
                [
                    [eps / 2, 1 - eps, eps / 2],
                    [eps / 2, 1 - eps, eps / 2],
                    [eps / 2, 1 - eps, eps / 2],
                ]
            )[None, :, :],
        ],
        dim=0,
    )
    e = torch.cat(
        [
            e[None, :, :],
            torch.tensor([[1 - eps, eps], [0.5, 0.5], [eps, 1 - eps]])[None, :, :],
        ],
        dim=0,
    )
    hmm_distr = MissingDataDiscreteHMM(torch.log(a0), torch.log(a), torch.log(e))
    states = hmm_distr.sample_states(x[1])
    expected_states = torch.tensor([[0, 1, 2, 1], [1, 1, 1, 1]])
    assert torch.allclose(states, expected_states)

    # Batch both parameters and values.
    states = hmm_distr.sample_states(x)
    expected_states = torch.tensor([[0, 1, 2, 2], [1, 1, 1, 1]])
    assert torch.allclose(states, expected_states)
