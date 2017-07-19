from __future__ import print_function

import torch
import time
import os
from os.path import isfile, join
from subprocess import Popen, PIPE, STDOUT

from tests.common import TestCase


class runExample(TestCase):
    def setUp(self):
        self.PATH='examples/'
        self.examples = [f for f in os.listdir(self.PATH) if isfile(join(self.PATH, f))]
        # __init___.py
        self.examples.pop(0)

    def test(self):
        for example in self.examples:
            print('running example ' + example + ' ... ', end='')
            cmd = 'python '+ self.PATH + example
            p = Popen(cmd, shell=True, stdin=PIPE,
                      stdout=PIPE, stderr=STDOUT, close_fds=True)
            output = p.stdout.read()
            if "Error" in output:
                self.fail(example + " threw an Error. Stack trace below:\n"
                          + output)
            else:
                print("ok")
