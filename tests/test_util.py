# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import warnings
import pytest

import torch
from pyro import util

pytestmark = pytest.mark.stage('unit')


def test_warn_if_nan():
    # scalar case
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = float('inf')
        msg = "example message"
        y = util.warn_if_nan(x, msg)
        assert y is x
        assert len(w) == 0

        x = float('nan')
        util.warn_if_nan(x, msg)
        # Verify some things
        assert len(w) == 1
        assert msg in str(w[-1].message)

    # tensor case
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = torch.ones(2)
        msg = "example message"
        util.warn_if_nan(x, msg)
        x[1] = float('nan')
        util.warn_if_nan(x, msg)
        assert len(w) == 1
        assert msg in str(w[-1].message)

    # grad case
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = torch.ones(2, requires_grad=True)
        util.warn_if_nan(x, msg)
        y = x.sum()
        y.backward([torch.tensor(float('nan'))])
        assert len(w) == 1
        assert msg in str(w[-1].message)


def test_warn_if_inf():
    # scalar case
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = 3
        msg = "example message"
        y = util.warn_if_inf(x, msg, allow_posinf=True, allow_neginf=True)
        assert y is x
        assert len(w) == 0
        x = float('inf')
        util.warn_if_inf(x, msg, allow_posinf=True)
        assert len(w) == 0
        util.warn_if_inf(x, msg, allow_neginf=True)
        assert len(w) == 1
        assert msg in str(w[-1].message)

    # tensor case
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = torch.ones(2)
        util.warn_if_inf(x, msg, allow_posinf=True, allow_neginf=True)
        assert len(w) == 0
        x[0] = float('inf')
        util.warn_if_inf(x, msg, allow_posinf=True)
        assert len(w) == 0
        util.warn_if_inf(x, msg, allow_neginf=True)
        assert len(w) == 1
        assert msg in str(w[-1].message)

    # grad case
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        x = torch.ones(2, requires_grad=True)
        util.warn_if_inf(x, msg, allow_posinf=True)
        y = x.sum()
        y.backward([torch.tensor(float('inf'))])
        assert len(w) == 0

        x.grad = None
        y.backward([torch.tensor(-float('inf'))])
        assert len(w) == 1
        assert msg in str(w[-1].message)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        z = torch.ones(2, requires_grad=True)
        y = z.sum()
        util.warn_if_inf(z, msg, allow_neginf=True)
        y.backward([torch.tensor(-float('inf'))])
        assert len(w) == 0
        z.grad = None
        y.backward([torch.tensor(float('inf'))])
        assert len(w) == 1
        assert msg in str(w[-1].message)
