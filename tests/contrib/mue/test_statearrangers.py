import torch

from pyro.contrib.mue.statearrangers import profile, mg2k
import pytest


def simpleprod(lst):
    # Product of list of scalar tensors, as numpy would do it.
    if len(lst) == 0:
        return torch.tensor(1.)
    else:
        return torch.prod(torch.cat([elem[None] for elem in lst]))


@pytest.mark.parametrize('M', [2, 20])
@pytest.mark.parametrize('batch_size', [None, 5])
@pytest.mark.parametrize('substitute', [False, True])
def test_profile(M, batch_size, substitute):
    torch.set_default_tensor_type('torch.DoubleTensor')

    pf_arranger = profile(M)

    u1 = torch.rand((M+1, 3))
    u = torch.cat([(1-u1)[:, :, None], u1[:, :, None]], dim=2)
    r1 = torch.rand((M+1, 3))
    r = torch.cat([(1-r1)[:, :, None], r1[:, :, None]], dim=2)
    s = torch.rand((M+1, 4))
    s = s/torch.sum(s, dim=1, keepdim=True)
    c = torch.rand((M+1, 4))
    c = c/torch.sum(c, dim=1, keepdim=True)

    if batch_size is not None:
        s = torch.rand((batch_size, M+1, 4))
        s = s/torch.sum(s, dim=2, keepdim=True)
        u1 = torch.rand((batch_size, M+1, 3))
        u = torch.cat([(1-u1)[:, :, :, None], u1[:, :, :, None]], dim=3)

    if substitute:
        ll = torch.rand((4, 5))
        ll = ll/torch.sum(ll, dim=1, keepdim=True)
        a0ln, aln, eln = pf_arranger.forward(torch.log(s), torch.log(c),
                                             torch.log(r), torch.log(u),
                                             torch.log(ll))
    else:
        a0ln, aln, eln = pf_arranger.forward(torch.log(s), torch.log(c),
                                             torch.log(r), torch.log(u))

    # - Remake transition matrices. -
    K = 2*(M+1)
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
    chk_a = torch.zeros((batch_dim_size, K, K))
    chk_a0 = torch.zeros((batch_dim_size, K))
    chk_e = torch.zeros((batch_dim_size, K, 4))
    for b in range(batch_dim_size):
        m, g = -1, 0
        u1[b][-1] = 1e-32
        for mp in range(M+1):
            for gp in range(2):
                kp = mg2k(mp, gp)
                if m + 1 - g == mp and gp == 0:
                    chk_a0[b, kp] = (1 - r1[b, m+1-g, g])*(1 - u1[b, m+1-g, g])
                elif m + 1 - g < mp and gp == 0:
                    chk_a0[b, kp] = (
                            (1 - r1[b, m+1-g, g]) * u1[b, m+1-g, g] *
                            simpleprod([(1 - r1[b, mpp, 2])*u1[b, mpp, 2]
                                        for mpp in
                                        range(m+2-g, mp)]) *
                            (1 - r1[b, mp, 2]) * (1 - u1[b, mp, 2]))
                elif m + 1 - g == mp and gp == 1:
                    chk_a0[b, kp] = r1[b, m+1-g, g]
                elif m + 1 - g < mp and gp == 1:
                    chk_a0[b, kp] = (
                            (1 - r1[b, m+1-g, g]) * u1[b, m+1-g, g] *
                            simpleprod([(1 - r1[b, mpp, 2])*u1[b, mpp, 2]
                                        for mpp in
                                        range(m+2-g, mp)]) * r1[b, mp, 2])
        for m in range(M+1):
            for g in range(2):
                k = mg2k(m, g)
                for mp in range(M+1):
                    for gp in range(2):
                        kp = mg2k(mp, gp)
                        if m + 1 - g == mp and gp == 0:
                            chk_a[b, k, kp] = (1 - r1[b, m+1-g, g]
                                               )*(1 - u1[b, m+1-g, g])
                        elif m + 1 - g < mp and gp == 0:
                            chk_a[b, k, kp] = (
                                    (1 - r1[b, m+1-g, g]) * u1[b, m+1-g, g] *
                                    simpleprod([(1 - r1[b, mpp, 2]) *
                                                u1[b, mpp, 2]
                                                for mpp in range(m+2-g, mp)]) *
                                    (1 - r1[b, mp, 2]) * (1 - u1[b, mp, 2]))
                        elif m + 1 - g == mp and gp == 1:
                            chk_a[b, k, kp] = r1[b, m+1-g, g]
                        elif m + 1 - g < mp and gp == 1:
                            chk_a[b, k, kp] = (
                                    (1 - r1[b, m+1-g, g]) * u1[b, m+1-g, g] *
                                    simpleprod([(1 - r1[b, mpp, 2]) *
                                                u1[b, mpp, 2]
                                                for mpp in
                                                range(m+2-g, mp)]
                                               ) * r1[b, mp, 2])
                        elif m == M and mp == M and g == 0 and gp == 0:
                            chk_a[b, k, kp] = 1.

        for m in range(M+1):
            for g in range(2):
                k = mg2k(m, g)
                if g == 0:
                    chk_e[b, k, :] = s[b, m, :]
                else:
                    chk_e[b, k, :] = c[b, m, :]
    if substitute:
        chk_e = torch.matmul(chk_e, ll)
    # - -
    if batch_size is None:
        chk_a = chk_a.squeeze()
        chk_a0 = chk_a0.squeeze()
        chk_e = chk_e.squeeze()

        assert torch.allclose(torch.sum(torch.exp(a0ln)), torch.tensor(1.),
                              atol=1e-3, rtol=1e-3)
        assert torch.allclose(torch.sum(torch.exp(aln), axis=1)[:-1],
                              torch.ones(2*(M+1)-1), atol=1e-3,
                              rtol=1e-3)
    assert torch.allclose(chk_a0, torch.exp(a0ln))
    assert torch.allclose(chk_a, torch.exp(aln))
    assert torch.allclose(chk_e, torch.exp(eln))
