from __future__ import print_function

import torch
import time
import os
import sys
import subprocess
from os.path import isfile, join
from subprocess import Popen, PIPE, STDOUT

from tests.common import TestCase


class runExample(TestCase):
    def setUp(self):
        self.PATH = 'examples/'
        self.examples = [f for f in os.listdir(self.PATH) if isfile(join(self.PATH, f))]
        # remove __init___.py
        self.examples.pop(0)
        self.num_epochs = 3

    def test(self):
        for example in self.examples:
            print('running example ' + example + ' ... ', end='')
            sys.stdout.flush()  # need to manually do this in Python 2.7
            cmd = 'python ' + self.PATH + example + ' -n ' + str(self.num_epochs)
            p = Popen(cmd, shell=True, stdin=PIPE,
                      stdout=PIPE, stderr=STDOUT, close_fds=True)
            output = p.stdout.read()
            if p.wait() == 0:
                print('ok')
            else:
                self.fail(example + ' threw an Error. Stack trace below:\n' + output)
        self.assertTrue(True)
