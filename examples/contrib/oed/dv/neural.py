import torch
from torch import nn


class T_neural(nn.Module):
    def __init__(self, design_dim, y_dim):
        super(T_neural, self).__init__()
        input_dim = design_dim + 2*y_dim + 1
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, design, trace, observation_label):
        trace.compute_log_prob()
        lp = trace.nodes[observation_label]["log_prob"]
        y = trace.nodes[observation_label]["value"]

        design_suff = design.sum(-2, keepdim=True)
        suff = torch.matmul(y.unsqueeze(-2), design)
        squares = torch.matmul((y**2).unsqueeze(-2), design)
        allsquare = suff**2
        lp_unsqueezed = lp.unsqueeze(-1).unsqueeze(-2)
        m = torch.cat([squares, allsquare, design_suff, lp_unsqueezed], -1)
        h1 = self.softplus(self.linear1(m))
        o = self.linear2(h1).squeeze(-2).squeeze(-1)
        return o


class T_specialized(nn.Module):
    def __init__(self):
        super(T_specialized, self).__init__()
        self.group1 = nn.Linear(3, 1)
        self.group2 = nn.Linear(3, 1)

    def forward(self, design, trace, observation_label):
        trace.compute_log_prob()
        lp = trace.nodes[observation_label]["log_prob"]
        y = trace.nodes[observation_label]["value"]

        design_lengths = design.sum(-2, keepdim=True)
        group_sums = torch.matmul(y.unsqueeze(-2), design)
        group_square_sums = torch.matmul((y**2).unsqueeze(-2), design)
        group_sum_squares = group_sums**2
        reweighted_group_sum_squares = group_sum_squares * design_lengths

        features = torch.stack([group_square_sums,
                                group_sum_squares,
                                reweighted_group_sum_squares], dim=-1)
        grp1 = self.group1(features[..., 0, :]).squeeze(-1).squeeze(-1)
        grp2 = self.group2(features[..., 1, :]).squeeze(-1).squeeze(-1)

        return lp - grp1 - grp2
