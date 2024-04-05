# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch


def hessian(y, xs):
    """
    Convenience method for computing hessians. Note that this is slow in high
    dimensions because computing hessians in a reverse-mode AD library like
    PyTorch is inherently slow (note the for loop).
    """
    dys = torch.autograd.grad(y, xs, create_graph=True)
    flat_dy = torch.cat([dy.reshape(-1) for dy in dys])
    H = []
    for dyi in flat_dy:
        Hi = torch.cat([Hij.reshape(-1) for Hij in torch.autograd.grad(dyi, xs, retain_graph=True)])
        H.append(Hi)
    H = torch.stack(H)
    return H
