import torch
from torch import nn


class DVNeuralNet(nn.Module):
    def __init__(self, design_dim, y_dim):
        super(DVNeuralNet, self).__init__()
        input_dim = design_dim + 2*y_dim + 1
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.linear3 = nn.Linear(input_dim, 1)
        self.softplus = nn.Softplus()

    def forward(self, design, y, lp):
        design_suff = design.sum(-2, keepdim=True)
        suff = torch.matmul(y.unsqueeze(-2), design)
        squares = torch.matmul((y**2).unsqueeze(-2), design)
        allsquare = suff**2
        lp_unsqueezed = lp.unsqueeze(-1).unsqueeze(-2)
        m = torch.cat([squares, allsquare, design_suff, lp_unsqueezed], -1)
        h1 = self.softplus(self.linear1(m))
        h2 = self.softplus(self.linear2(h1))
        o = self.linear3(h2).squeeze(-2).squeeze(-1)
        return o
