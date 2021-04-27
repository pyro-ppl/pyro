import torch


def get_mb_indices(N_data, mini_batch_size):
    extra = N_data % mini_batch_size
    N_mb = int(N_data / mini_batch_size) + int(bool(extra))
    shuffled_indices = torch.randperm(N_data)

    if extra > 0:
        shuffled_indices = torch.cat([shuffled_indices, torch.zeros(mini_batch_size - extra).type_as(shuffled_indices)])
        masks = [torch.ones(mini_batch_size).bool() for k in range(N_mb - 1)]
        masks.append(torch.cat([torch.ones(extra), torch.zeros(mini_batch_size - extra)]).bool())
    else:
        masks = [torch.ones(mini_batch_size).bool() for k in range(N_mb)]
    mb_indices = [shuffled_indices[k * mini_batch_size: (k+1) * mini_batch_size] for k in range(N_mb)]

    return mb_indices, masks
