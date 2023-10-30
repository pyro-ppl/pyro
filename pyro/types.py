# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import torch
from typing_extensions import TypedDict

from pyro.poutine.indep_messenger import CondIndepStackFrame


class Message(TypedDict, total=False):
    type: Optional[str]
    name: str
    fn: Callable
    is_observed: bool
    args: Tuple
    kwargs: Dict
    value: Optional[torch.Tensor]
    scale: float
    mask: Union[bool, torch.Tensor, None]
    cond_indep_stack: Tuple[CondIndepStackFrame, ...]
    done: bool
    stop: bool
    continuation: Optional[Callable[[Message], None]]
    infer: Optional[Dict[str, Union[str, bool]]]
    obs: Optional[torch.Tensor]
