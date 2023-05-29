# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pyro.contrib.mue.statearrangers import Profile, mg2k


def simpleprod(lst):
    # Product of list of scalar tensors, as numpy would do it.
    if len(lst) == 0:
        return torch.tensor(1.0)
    else:
        return torch.prod(torch.cat([elem[None] for elem in lst]))


@pytest.mark.parametrize("M", [2, 20])
@pytest.mark.parametrize("batch_size", [None, 5])
@pytest.mark.parametrize("substitute", [False, True])
def test_profile_alternate_imp(M, batch_size, substitute):
    # --- Setup random model. ---
    pf_arranger = Profile(M)

    u1 = torch.rand((M + 1, 3))
    u1[M, :] = 0  # Assume u_{M+1, j} = 0 for j in {0, 1, 2} in Eqn. S40.
    u = torch.cat([(1 - u1)[:, :, None], u1[:, :, None]], dim=2)
    r1 = torch.rand((M + 1, 3))
    r1[M, :] = 1  # Assume r_{M+1, j} = 1 for j in {0, 1, 2} in Eqn. S40.
    r = torch.cat([(1 - r1)[:, :, None], r1[:, :, None]], dim=2)
    s = torch.rand((M, 4))
    s = s / torch.sum(s, dim=1, keepdim=True)
    c = torch.rand((M + 1, 4))
    c = c / torch.sum(c, dim=1, keepdim=True)

    if batch_size is not None:
        s = torch.rand((batch_size, M, 4))
        s = s / torch.sum(s, dim=2, keepdim=True)
        u1 = torch.rand((batch_size, M + 1, 3))
        u1[:, M, :] = 0
        u = torch.cat([(1 - u1)[:, :, :, None], u1[:, :, :, None]], dim=3)

    # Compute forward pass of state arranger to get HMM parameters.
    # Don't use dimension M, assumed fixed by statearranger.
    if substitute:
        ll = torch.rand((4, 5))
        ll = ll / torch.sum(ll, dim=1, keepdim=True)
        a0ln, aln, eln = pf_arranger.forward(
            torch.log(s),
            torch.log(c),
            torch.log(r[:-1, :]),
            torch.log(u[..., :-1, :, :]),
            torch.log(ll),
        )
    else:
        a0ln, aln, eln = pf_arranger.forward(
            torch.log(s),
            torch.log(c),
            torch.log(r[:-1, :]),
            torch.log(u[..., :-1, :, :]),
        )

    # - Remake HMM parameters to check. -
    # Here we implement Equation S40 from the MuE paper
    # (https://www.biorxiv.org/content/10.1101/2020.07.31.231381v1.full.pdf)
    # more directly, iterating over all the indices of the transition matrix
    # and initial transition vector.
    K = 2 * M + 1
    if batch_size is None:
        batch_dim_size = 1
        r1 = r1.unsqueeze(0)
        u1 = u1.unsqueeze(0)
        s = s.unsqueeze(0)
        c = c.unsqueeze(0)
        if substitute:
            ll = ll.unsqueeze(0)
    else:
        batch_dim_size = batch_size
        r1 = r1[None, :, :] * torch.ones([batch_size, 1, 1])
        c = c[None, :, :] * torch.ones([batch_size, 1, 1])
        if substitute:
            ll = ll.unsqueeze(0)
    expected_a = torch.zeros((batch_dim_size, K, K))
    expected_a0 = torch.zeros((batch_dim_size, K))
    expected_e = torch.zeros((batch_dim_size, K, 4))
    for b in range(batch_dim_size):
        m, g = -1, 0
        u1[b][-1] = 1e-32
        for gp in range(2):
            for mp in range(M + gp):
                kp = mg2k(mp, gp, M)
                if m + 1 - g == mp and gp == 0:
                    expected_a0[b, kp] = (1 - r1[b, m + 1 - g, g]) * (
                        1 - u1[b, m + 1 - g, g]
                    )
                elif m + 1 - g < mp and gp == 0:
                    expected_a0[b, kp] = (
                        (1 - r1[b, m + 1 - g, g])
                        * u1[b, m + 1 - g, g]
                        * simpleprod(
                            [
                                (1 - r1[b, mpp, 2]) * u1[b, mpp, 2]
                                for mpp in range(m + 2 - g, mp)
                            ]
                        )
                        * (1 - r1[b, mp, 2])
                        * (1 - u1[b, mp, 2])
                    )
                elif m + 1 - g == mp and gp == 1:
                    expected_a0[b, kp] = r1[b, m + 1 - g, g]
                elif m + 1 - g < mp and gp == 1:
                    expected_a0[b, kp] = (
                        (1 - r1[b, m + 1 - g, g])
                        * u1[b, m + 1 - g, g]
                        * simpleprod(
                            [
                                (1 - r1[b, mpp, 2]) * u1[b, mpp, 2]
                                for mpp in range(m + 2 - g, mp)
                            ]
                        )
                        * r1[b, mp, 2]
                    )
        for g in range(2):
            for m in range(M + g):
                k = mg2k(m, g, M)
                for gp in range(2):
                    for mp in range(M + gp):
                        kp = mg2k(mp, gp, M)
                        if m + 1 - g == mp and gp == 0:
                            expected_a[b, k, kp] = (1 - r1[b, m + 1 - g, g]) * (
                                1 - u1[b, m + 1 - g, g]
                            )
                        elif m + 1 - g < mp and gp == 0:
                            expected_a[b, k, kp] = (
                                (1 - r1[b, m + 1 - g, g])
                                * u1[b, m + 1 - g, g]
                                * simpleprod(
                                    [
                                        (1 - r1[b, mpp, 2]) * u1[b, mpp, 2]
                                        for mpp in range(m + 2 - g, mp)
                                    ]
                                )
                                * (1 - r1[b, mp, 2])
                                * (1 - u1[b, mp, 2])
                            )
                        elif m + 1 - g == mp and gp == 1:
                            expected_a[b, k, kp] = r1[b, m + 1 - g, g]
                        elif m + 1 - g < mp and gp == 1:
                            expected_a[b, k, kp] = (
                                (1 - r1[b, m + 1 - g, g])
                                * u1[b, m + 1 - g, g]
                                * simpleprod(
                                    [
                                        (1 - r1[b, mpp, 2]) * u1[b, mpp, 2]
                                        for mpp in range(m + 2 - g, mp)
                                    ]
                                )
                                * r1[b, mp, 2]
                            )
                        elif m == M and mp == M and g == 0 and gp == 0:
                            expected_a[b, k, kp] = 1.0

        for g in range(2):
            for m in range(M + g):
                k = mg2k(m, g, M)
                if g == 0:
                    expected_e[b, k, :] = s[b, m, :]
                else:
                    expected_e[b, k, :] = c[b, m, :]
    if substitute:
        expected_e = torch.matmul(expected_e, ll)

    # --- Check ---
    if batch_size is None:
        expected_a = expected_a.squeeze()
        expected_a0 = expected_a0.squeeze()
        expected_e = expected_e.squeeze()

        assert torch.allclose(
            torch.sum(torch.exp(a0ln)), torch.tensor(1.0), atol=1e-3, rtol=1e-3
        )
        assert torch.allclose(
            torch.sum(torch.exp(aln), axis=1),
            torch.ones(2 * M + 1),
            atol=1e-3,
            rtol=1e-3,
        )
    assert torch.allclose(expected_a0, torch.exp(a0ln))
    assert torch.allclose(expected_a, torch.exp(aln))
    assert torch.allclose(expected_e, torch.exp(eln))


@pytest.mark.parametrize("batch_ancestor_seq", [False, True])
@pytest.mark.parametrize("batch_insert_seq", [False, True])
@pytest.mark.parametrize("batch_insert", [False, True])
@pytest.mark.parametrize("batch_delete", [False, True])
@pytest.mark.parametrize("batch_substitute", [False, True])
def test_profile_shapes(
    batch_ancestor_seq, batch_insert_seq, batch_insert, batch_delete, batch_substitute
):
    M, D, B = 5, 2, 3
    K = 2 * M + 1
    batch_size = 6
    pf_arranger = Profile(M)
    sln = torch.randn([batch_size] * batch_ancestor_seq + [M, D])
    sln = sln - sln.logsumexp(-1, True)
    cln = torch.randn([batch_size] * batch_insert_seq + [M + 1, D])
    cln = cln - cln.logsumexp(-1, True)
    rln = torch.randn([batch_size] * batch_insert + [M, 3, 2])
    rln = rln - rln.logsumexp(-1, True)
    uln = torch.randn([batch_size] * batch_delete + [M, 3, 2])
    uln = uln - uln.logsumexp(-1, True)
    lln = torch.randn([batch_size] * batch_substitute + [D, B])
    lln = lln - lln.logsumexp(-1, True)
    a0ln, aln, eln = pf_arranger.forward(sln, cln, rln, uln, lln)

    if all([not batch_ancestor_seq, not batch_insert_seq, not batch_substitute]):
        assert eln.shape == (K, B)
        assert torch.allclose(eln.logsumexp(-1), torch.zeros(K))
    else:
        assert eln.shape == (batch_size, K, B)
        assert torch.allclose(eln.logsumexp(-1), torch.zeros(batch_size, K))

    if all([not batch_insert, not batch_delete]):
        assert a0ln.shape == (K,)
        assert torch.allclose(a0ln.logsumexp(-1), torch.zeros(1))
        assert aln.shape == (K, K)
        assert torch.allclose(aln.logsumexp(-1), torch.zeros(K))
    else:
        assert a0ln.shape == (batch_size, K)
        assert torch.allclose(a0ln.logsumexp(-1), torch.zeros(batch_size))
        assert aln.shape == (batch_size, K, K)
        assert torch.allclose(aln.logsumexp(-1), torch.zeros((batch_size, K)))


@pytest.mark.parametrize("M", [2, 20])  # , 20
def test_profile_trivial_cases(M):
    # Trivial case: indel probabability of zero. Expected value of
    # HMM should match ancestral sequence times substitution matrix.

    # --- Setup model. ---
    D, B = 2, 2
    batch_size = 5
    pf_arranger = Profile(M)
    sln = torch.randn([batch_size, M, D])
    sln = sln - sln.logsumexp(-1, True)
    cln = torch.randn([batch_size, M + 1, D])
    cln = cln - cln.logsumexp(-1, True)
    rln = torch.cat(
        [torch.zeros([M, 3, 1]), -1 / pf_arranger.epsilon * torch.ones([M, 3, 1])],
        axis=-1,
    )
    uln = torch.cat(
        [torch.zeros([M, 3, 1]), -1 / pf_arranger.epsilon * torch.ones([M, 3, 1])],
        axis=-1,
    )
    lln = torch.randn([D, B])
    lln = lln - lln.logsumexp(-1, True)

    a0ln, aln, eln = pf_arranger.forward(sln, cln, rln, uln, lln)

    # --- Compute expected value per step. ---
    Eyln = torch.zeros([batch_size, M, B])
    ai = a0ln
    for j in range(M):
        Eyln[:, j, :] = torch.logsumexp(ai.unsqueeze(-1) + eln, axis=-2)
        ai = torch.logsumexp(ai.unsqueeze(-1) + aln, axis=-2)

    print(aln.exp())
    no_indel = torch.logsumexp(sln.unsqueeze(-1) + lln.unsqueeze(-3), axis=-2)
    assert torch.allclose(Eyln, no_indel)
