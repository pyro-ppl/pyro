from __future__ import absolute_import, division, print_function

import numpy as np
import torch

J = 8
y = np.array([28,  8, -3,  7, -1,  1, 18, 12])
sigma = np.array([15, 10, 16, 11,  9, 11, 10, 18])

y_tensor = torch.tensor(y).type(torch.Tensor)
sigma_tensor = torch.tensor(sigma).type(torch.Tensor)
