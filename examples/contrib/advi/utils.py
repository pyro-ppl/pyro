import json

import torch


def get_data(fname, varnames):
    with open(fname, "r") as f:
        j = json.load(f)
    d = {}
    for i in range(len(j[0])):
        var_name = j[0][i]
        if isinstance(j[1][i], int):
            val = j[1][i]
        else:
            val = torch.tensor(j[1][i])
        d[var_name] = val
    return ([d[k] for k in varnames])
