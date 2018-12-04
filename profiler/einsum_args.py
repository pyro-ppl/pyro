#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

import glob
import os
import subprocess
from collections import Counter

import six.moves.cPickle as pickle


EINSUM_COUNTS = Counter()


def print_counts():
    print('COUNT\tEQUATION\tSHAPE1\tSHAPE2')
    for cols, count in EINSUM_COUNTS.most_common():
        if len(cols) == 3:
            print('{}\t{}\t{}\t{}'.format(count, *cols))


if __name__ == '__main__':
    for filename in glob.glob('/tmp/pyro.einsum.*'):
        os.remove(filename)
    subprocess.check_call('PYTHONSTARTUP={} '
                          'pytest '
                          '-vx '
                          '-n auto '
                          'tests/infer/test_valid_models.py '
                          'tests/infer/test_enum.py '
                          'tests/test_examples.py::test_cpu '
                          ''.format(os.path.abspath(__file__)), shell=True)
    for filename in glob.glob('/tmp/pyro.einsum.*'):
        with open(filename, 'rb') as f:
            counts = pickle.load(f)
            EINSUM_COUNTS.update(counts)
    print_counts()
