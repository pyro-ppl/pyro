from __future__ import absolute_import, division, print_function

import logging

logger = logging.getLogger(__name__)

# create log handler for tests
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname).1s %(name)s \t %(message)s')
handler.setFormatter(formatter)

# set default logging level for tests
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
