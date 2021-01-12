# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import itertools

import numpy as np
import scipy

from pyro.distributions.testing.special import chi2sf
from tests.common import assert_close


def test_chi2sf():
    xlist = np.linspace(0, 100, 500)
    slist = np.arange(1, 41, 1.5)
    for s, x in itertools.product(slist, xlist):
        assert_close(scipy.stats.chi2.sf(x, s), chi2sf(x, s))
