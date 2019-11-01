import pytest
import torch

import pyro.distributions as dist


@pytest.mark.parametrize('peak', [0.1, 0.3, 0.5, 0.7, 0.9])
def test_simulate_uniform(peak):
    n_samples = 10 ** 6
    x = torch.rand(len(peak))

    u = torch.FloatTensor(n_samples).uniform_()
    v = torch.FloatTensor(n_samples).uniform_()
    # From William E. Stein and Matthew F. Keblis
    # "A new method to simulate the triangular distribution."
    # Mathematical and Computer Modelling, Volume 49, Issues 5–6, March 2009, Pages 1143-1147
    sim_triangular = peak + (u[:, None] - peak) * v[:, None].sqrt()
    sim_prob = (sim_triangular < x).sum(0) / (n_samples * 1.)

    triangular = dist.Triangular(0., 1., peak)
    prob = triangular.cdf(x)

    assert torch.all(torch.abs(prob - sim_prob) < 1e-3)
