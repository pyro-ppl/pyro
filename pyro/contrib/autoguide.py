# Copyright (c) 2017-2020 Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import warnings

from pyro.infer.autoguide import *  # noqa F403

warnings.warn("pyro.contrib.autoguide has moved to pyro.infer.autoguide. "
              "The contrib alias will stop working in Pyro 0.5.",
              DeprecationWarning)
