import torch.nn as nn
from torch.nn.functional import sigmoid, softplus


# Takes pixel intensities of the attention window to parameters (mean,
# standard deviation) of the distribution over the latent code,
# z_what.
class Encoder(nn.Module):
    def __init__(self, x_size, h_sizes, z_size, non_linear_layer):
        super(Encoder, self).__init__()
        self.z_size = z_size
        output_size = 2 * z_size
        self.mlp = MLP(x_size, h_sizes + [output_size], non_linear_layer)

    def forward(self, x):
        a = self.mlp(x)
        return a[:, 0:self.z_size], softplus(a[:, self.z_size:])


# Takes a latent code, z_what, to pixel intensities.
class Decoder(nn.Module):
    def __init__(self, x_size, h_sizes, z_size, bias, use_sigmoid, non_linear_layer):
        super(Decoder, self).__init__()
        self.bias = bias
        self.use_sigmoid = use_sigmoid
        self.mlp = MLP(z_size, h_sizes + [x_size], non_linear_layer)

    def forward(self, z):
        a = self.mlp(z)
        if self.bias is not None:
            a = a + self.bias
        return sigmoid(a) if self.use_sigmoid else a


# A general purpose module to construct networks that look like:
# [Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1)]
# [Linear (256 -> 256), ReLU (), Linear (256 -> 1), ReLU ()]
# etc.
class MLP(nn.Module):
    def __init__(self, in_size, out_sizes, non_linear_layer, output_non_linearity=False):
        super(MLP, self).__init__()
        assert len(out_sizes) >= 1
        layers = []
        in_sizes = [in_size] + out_sizes[0:-1]
        sizes = list(zip(in_sizes, out_sizes))
        for (i, o) in sizes[0:-1]:
            layers.append(nn.Linear(i, o))
            layers.append(non_linear_layer())
        layers.append(nn.Linear(sizes[-1][0], sizes[-1][1]))
        if output_non_linearity:
            layers.append(non_linear_layer())
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


# Takes the guide RNN hidden state to parameters of the guide
# distributions over z_where and z_pres.
class Predict(nn.Module):
    def __init__(self, input_size, h_sizes, z_pres_size, z_where_size, non_linear_layer):
        super(Predict, self).__init__()
        self.z_pres_size = z_pres_size
        self.z_where_size = z_where_size
        output_size = z_pres_size + 2 * z_where_size
        self.mlp = MLP(input_size, h_sizes + [output_size], non_linear_layer)

    def forward(self, h):
        out = self.mlp(h)
        z_pres_p = sigmoid(out[:, 0:self.z_pres_size])
        z_where_loc = out[:, self.z_pres_size:self.z_pres_size + self.z_where_size]
        z_where_scale = softplus(out[:, (self.z_pres_size + self.z_where_size):])
        return z_pres_p, z_where_loc, z_where_scale


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
