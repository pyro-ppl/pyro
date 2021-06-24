# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings


def test_warnings():
    with warnings.catch_warnings(record=True) as w1:
        with warnings.catch_warnings(record=True) as w2:
            warnings.filterwarnings("ignore")
            warnings.filterwarnings("always", "foo", category=RuntimeWarning)
            warnings.warn("foo", RuntimeWarning)
            warnings.warn("bar", RuntimeWarning)
    print(f"w1 = {w1}")
    print(f"w2 = {w2}")
