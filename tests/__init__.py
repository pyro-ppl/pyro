from __future__ import absolute_import, division, print_function

import logging

# create log handler for tests
import os

level = logging.INFO if "CI" in os.environ else logging.DEBUG
logging.basicConfig(format='%(levelname).1s \t %(message)s', level=level)
