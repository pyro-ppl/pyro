import torch

from pyro.contrib.mue import profile, mg2k
import pytest


@pytest.mark.parameterize('M', [2, 20])
@pytest.mark.parameterize('batch_size', [None])
@pytest.mark.parameterize('substitute', [False, True])
def test_profile(M, batch_size):
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
        s = torch.ones([batch_size, 1, 1]) * s[None, :, :]
        u = torch.ones([batch_size, 1, 1]) * u[None, :, :]

    a0ln, aln, eln = pf_arranger.forward(torch.log(s), torch.log(c),
                                         torch.log(r), torch.log(u))

    # - Remake transition matrices. -
    u1[-1] = 1e-32
    K = 2*(M+1)
    chk_a = torch.zeros((K, K))
    chk_a0 = torch.zeros((K,))
    m, g = -1, 0
    for mp in range(M+1):
        for gp in range(2):
            kp = mg2k(mp, gp)
            if m + 1 - g == mp and gp == 0:
                chk_a0[kp] = (1 - r1[m+1-g])*(1 - u1[m+1-g])
            elif m + 1 - g < mp and gp == 0:
                chk_a0[kp] = (
                        (1 - r1[m+1-g]) * u1[m+1-g] *
                        torch.prod([(1 - r1[mpp])*u1[mpp] for mpp in
                                    range(m+2-g, mp)]) *
                        (1 - r1[mp]) * (1 - u1[mp]))
            elif m + 1 - g == mp and gp == 1:
                chk_a0[kp] = r1[m+1-g]
            elif m + 1 - g < mp and gp == 1:
                chk_a0[kp] = (
                        (1 - r1[m+1-g]) * u1[m+1-g] *
                        torch.prod([(1 - r1[mpp])*u1[mpp] for mpp in
                                    range(m+2-g, mp)]) * r1[mp])
    for m in range(M+1):
        for g in range(2):
            k = mg2k(m, g)
            for mp in range(M+1):
                for gp in range(2):
                    kp = mg2k(mp, gp)
                    if m + 1 - g == mp and gp == 0:
                        chk_a[k, kp] = (1 - r1[m+1-g])*(1 - u1[m+1-g])
                    elif m + 1 - g < mp and gp == 0:
                        chk_a[k, kp] = (
                                (1 - r1[m+1-g]) * u1[m+1-g] *
                                torch.prod([(1 - r1[mpp])*u1[mpp] for mpp in
                                            range(m+2-g, mp)]) *
                                (1 - r1[mp]) * (1 - u1[mp]))
                    elif m + 1 - g == mp and gp == 1:
                        chk_a[k, kp] = r1[m+1-g]
                    elif m + 1 - g < mp and gp == 1:
                        chk_a[k, kp] = (
                                (1 - r1[m+1-g]) * u1[m+1-g] *
                                torch.prod([(1 - r1[mpp])*u1[mpp] for mpp in
                                            range(m+2-g, mp)]) * r1[mp])
                    elif m == M and mp == M and g == 0 and gp == 0:
                        chk_a[k, kp] = 1.

    chk_e = torch.zeros((2*(M+1), 4))
    for m in range(M+1):
        for g in range(2):
            k = mg2k(m, g)
            if g == 0:
                chk_e[k, :] = s[m, :].numpy()
            else:
                chk_e[k, :] = c[m, :].numpy()
    # - -

    assert torch.allclose(chk_a0, torch.exp(a0ln))
    assert torch.allclose(chk_a, torch.exp(aln))
    assert torch.allclose(chk_e, torch.exp(eln))

    # Check normalization.
    assert torch.allclose(torch.sum(torch.exp(a0ln)), 1., atol=1e-3,
                          rtol=1e-3)
    assert torch.allclose(torch.sum(torch.exp(aln), axis=1)[:-1],
                          torch.ones(2*(M+1)-1), atol=1e-3,
                          rtol=1e-3)
