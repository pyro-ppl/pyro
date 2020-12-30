import torch

from pyro.contrib.mue.variablelengthhmm import VariableLengthDiscreteHMM


def test_hmm_log_prob():
    torch.set_default_tensor_type('torch.DoubleTensor')

    a0 = torch.tensor([0.9, 0.08, 0.02])
    a = torch.tensor([[0.1, 0.8, 0.1], [0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
    e = torch.tensor([[0.99, 0.01], [0.01, 0.99], [0.5, 0.5]])

    x = torch.tensor([[0., 1.],
                      [1., 0.],
                      [0., 1.],
                      [0., 1.],
                      [1., 0.],
                      [0., 0.]])

    hmm_distr = VariableLengthDiscreteHMM(torch.log(a0), torch.log(a),
                                          torch.log(e))
    lp = hmm_distr.log_prob(x)

    f = torch.matmul(a0, a) * e[:, 1]
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

    f = torch.matmul(a0, a) * e[:, 0]
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
    hmm_distr = VariableLengthDiscreteHMM(torch.log(a0), torch.log(a),
                                          torch.log(e))
    lp = hmm_distr.log_prob(x)

    f = torch.matmul(a0[1, :], a[1, :, :]) * e[1, :, 0]
    f = torch.matmul(f, a[1, :, :]) * e[1, :, 0]
    f = torch.matmul(f, a[1, :, :]) * e[1, :, 0]
    chk_lp = torch.cat([chk_lp[0][None], torch.log(torch.sum(f))[None]])

    assert torch.allclose(lp, chk_lp)
