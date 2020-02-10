# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging


default_format = '%(levelname)s \t %(message)s'
log = logging.getLogger("pyro")
log.setLevel(logging.INFO)


if not logging.root.handlers:
    default_handler = logging.StreamHandler()
    default_handler.setLevel(logging.INFO)
    default_handler.setFormatter(logging.Formatter(default_format))
    log.addHandler(default_handler)
    log.propagate = False
