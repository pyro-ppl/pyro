# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch


J = 8
y = torch.tensor([28,  8, -3,  7, -1,  1, 18, 12]).type(torch.Tensor)
sigma = torch.tensor([15, 10, 16, 11,  9, 11, 10, 18]).type(torch.Tensor)
