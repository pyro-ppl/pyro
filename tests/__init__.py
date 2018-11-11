from __future__ import absolute_import, division, print_function

import logging

import os

# create log handler for tests
level = logging.INFO if "CI" in os.environ else logging.DEBUG
logging.basicConfig(format='%(levelname).1s \t %(message)s', level=level)
