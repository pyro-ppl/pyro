import sys
import logging
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


def get_logger(log_dir, log_file):

    formatter = logging.Formatter('%(message)s')

    def setup_logger(name, log_file, level=logging.DEBUG):
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

    if log_dir[-1] != '/':
        log_dir += '/'

    try:
        nfs_logger = setup_logger('nfs_logger', log_dir + log_file)
    except Exception as e:
        print("ERROR!!! Probably missing expected log directory: {}".format(log_dir))
        print(e)

    def log(s):
        if log_dir != '':
            nfs_logger.info(s)
        print(s)
        sys.stdout.flush()

    return log

