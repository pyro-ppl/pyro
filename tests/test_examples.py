import os
import glob
import sys
from subprocess import check_call

from tests.common import EXAMPLES_DIR
import pytest

EXAMPLES = [
    os.path.basename(f)
    for f in glob.glob(os.path.join(EXAMPLES_DIR, '*.py'))
    if not f.endswith('__init__.py')
]


@pytest.mark.parametrize('example', EXAMPLES)
def test_example(example):
    example = os.path.join(EXAMPLES_DIR, example)
    check_call([sys.executable, example, '-n', '1'])
