import torch.nn as nn
import torch.nn.functional as F


# Attention window encoder/decoder.
class Encoder(nn.Module):
    def __init__(self, x_size, z_size, h_size):
        super(Encoder, self).__init__()
        self.f = nn.Linear(x_size, h_size)
        self.g1 = nn.Linear(h_size, z_size)
        self.g2 = nn.Linear(h_size, z_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.f(x))
        return self.g1(h), F.softplus(self.g2(h))


class Decoder(nn.Module):
    def __init__(self, x_size, z_size, h_size, bias):
        super(Decoder, self).__init__()
        self.bias = bias
        self.f = nn.Linear(z_size, h_size)
        self.g = nn.Linear(h_size, x_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = self.relu(self.f(z))
        a = self.g(h)
        if self.bias is not None:
            a = a + self.bias
        return self.sigmoid(a)

# Takes the output of the rnn to parameters for guide distributions
# for z_where and z_pres.

# Makes modules that look like:
# [Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1)]
# etc.


class MLP(nn.Module):
    def __init__(self, in_size, out_size, hid_size, num_hid):
        super(MLP, self).__init__()
        assert type(num_hid) == int and num_hid >= 0
        layers = []
        for i in range(num_hid + 1):
            in_size_i = in_size if i == 0 else hid_size
            out_size_i = out_size if i == num_hid else hid_size
            layers.append(nn.Linear(in_size_i, out_size_i))
            if i < num_hid:
                layers.append(nn.ReLU())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class Predict(nn.Module):
    def __init__(self, input_size, hid_size, num_hid_layers, z_pres_size, z_where_size):
        super(Predict, self).__init__()
        self.z_pres_size = z_pres_size
        self.z_where_size = z_where_size
        self.mlp = MLP(input_size,
                       z_pres_size + 2 * z_where_size,
                       hid_size,
                       num_hid_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        out = self.mlp(h)
        z_pres_p = self.sigmoid(out[:, 0:self.z_pres_size])
        z_where_mu = out[:, self.z_pres_size:self.z_pres_size + self.z_where_size]
        z_where_sigma = F.softplus(out[:, (self.z_pres_size + self.z_where_size):])
        return z_pres_p, z_where_mu, z_where_sigma


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
