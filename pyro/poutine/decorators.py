from __future__ import absolute_import, division, print_function

import functools
from six.moves import xrange

from pyro.poutine import util

from .block_poutine import BlockMessenger
from .condition_poutine import ConditionMessenger
from .enumerate_poutine import EnumerateMessenger  # noqa: F401
from .escape_poutine import EscapeMessenger
from .indep_poutine import IndepMessenger  # noqa: F401
from .infer_config_poutine import InferConfigMessenger
from .lift_poutine import LiftMessenger
from .poutine import _PYRO_STACK, Messenger  # noqa: F401
from .replay_poutine import ReplayMessenger
from .scale_poutine import ScaleMessenger
from .trace import Trace  # noqa: F401
from .trace_poutine import TraceMessenger


block = BlockMessenger
