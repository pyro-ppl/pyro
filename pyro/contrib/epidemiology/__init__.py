# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from .compartmental import CompartmentalModel
from .sir import SIRModel

__all__ = [
    "CompartmentalModel",
    "SIRModel",
]
