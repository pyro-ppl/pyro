import torch
from torch import nn

import pyro
import pyro.distributions as dist


class Ba_lm_guide(nn.Module):

    def __init__(self, coef_shape):
        super(Ba_lm_guide, self).__init__()
        self.regu = nn.Parameter(10.*torch.ones(coef_shape[-1]))
        self.sds = nn.Parameter(10.*torch.ones(coef_shape))

    def forward(self, y, design):
        # design_suff = design.sum(-2, keepdim=True)
        # suff = torch.matmul(y.unsqueeze(-2), design)
        xtxi = tensorized_2_by_2_matrix_inverse(torch.matmul(design.transpose(-1, -2), design) + torch.diag(self.regu))
        mu = torch.matmul(xtxi, torch.matmul(design.transpose(-1, -2), y.unsqueeze(-1))).squeeze(-1)
        # mu = (suff/(design_suff + self.regu)).squeeze(-2)

        return mu, self.sds

    def guide(self, y_dict, design, observation_labels, target_labels):

        target_label = target_labels[0]

        pyro.module("ba_guide", self)

        y = y_dict["y"]
        # design is size batch x n x p
        # tau is size batch
        tau_shape = design.shape[:-2]

        # response will be shape batch x n

        # Set up mu and lambda
        mu, sigma = self.forward(y, design)
        
        # guide distributions for w
        w_dist = dist.Normal(mu, sigma).independent(1)
        w = pyro.sample(target_label, w_dist)


def tensorized_2_by_2_matrix_inverse(M):
    det = M[..., 0, 0]*M[..., 1, 1] - M[..., 1, 0]*M[..., 0, 1]
    inv = torch.zeros(M.shape)
    inv[..., 0, 0] = M[..., 1, 1]
    inv[..., 1, 1] = M[..., 0, 0]
    inv[..., 0, 1] = -M[..., 0, 1]
    inv[..., 1, 0] = -M[..., 1, 0]
    inv = inv/det.unsqueeze(-1).unsqueeze(-1)
    return inv
