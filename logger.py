import logging
import sys


def get_logger(log_dir, log_file, use_local_logger=True):

    formatter = logging.Formatter('%(message)s')

    def setup_logger(name, log_file, level=logging.DEBUG):
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

    if use_local_logger:
        local_logger = setup_logger('local_logger', log_file)

    if log_dir != '':
        if log_dir[-1] != '/':
            log_dir += '/'
        nfs_logger = setup_logger('nfs_logger', log_dir + log_file)

    def log(s):
        if use_local_logger:
            local_logger.info(s)
        if log_dir != '':
            nfs_logger.info(s)
        print(s)
        sys.stdout.flush()

    return log
