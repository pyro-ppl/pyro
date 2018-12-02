import sys
import logging
import torch


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
    nfs_logger = setup_logger('nfs_logger', log_dir + log_file)

    def log(s):
        if log_dir != '':
            nfs_logger.info(s)
        print(s)
        sys.stdout.flush()

    return log


def get_mb_indices(N_data, mini_batch_size):
    N_mb = int(N_data / mini_batch_size) + int(bool(N_data % mini_batch_size))
    shuffled_indices = torch.randperm(N_data)
    mb_indices = []
    for k in range(N_mb):
        mb_indices.append(shuffled_indices[k * mini_batch_size: min((k+1) * mini_batch_size, N_data)])
    return mb_indices
