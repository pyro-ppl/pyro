# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.distributions as dist

from pyro.distributions.testing.gof import multinomial_goodness_of_fit


def test_multinomial_goodness_of_fit():
    N = 100000
    K = 20
    logits = torch.randn(K)
    probs = (logits - logits.logsumexp(-1)).exp()
    d = dist.Categorical(probs)
    samples = d.sample((N,))
    counts = torch.zeros(K, dtype=torch.long)
    counts.scatter_add_(0, samples, torch.ones(N, dtype=torch.long))
    gof = multinomial_goodness_of_fit(probs, counts, plot=True)
    assert gof > 0.1
