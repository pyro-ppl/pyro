# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
VERSION = """
# This file is auto-generated with the version information during setup.py installation.

__version__ = '{}'
"""

# Find pyro version.
for line in open(os.path.join(PROJECT_PATH, 'pyro', '__init__.py')):
    if line.startswith('version_prefix = '):
        version = line.strip().split()[2][1:-1]

# Append current commit sha to version
commit_sha = ''
try:
    current_tag = subprocess.check_output(['git', 'tag', '--points-at', 'HEAD'],
                                          cwd=PROJECT_PATH).decode('ascii').strip()
    # only add sha if HEAD does not point to the release tag
    if not current_tag == version:
        commit_sha = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                             cwd=PROJECT_PATH).decode('ascii').strip()
# catch all exception to be safe
except Exception:
    pass  # probably not a git repo

# Write version to _version.py
if commit_sha:
    version += '+{}'.format(commit_sha)
with open(os.path.join(PROJECT_PATH, 'pyro', '_version.py'), 'w') as f:
    f.write(VERSION.format(version))


# READ README.md for long description on PyPi.
# This requires uploading via twine, e.g.:
# $ python setup.py sdist bdist_wheel
# $ twine upload --repository-url https://test.pypi.org/legacy/ dist/*  # test version
# $ twine upload dist/*
try:
    long_description = open('README.md', encoding='utf-8').read()
except Exception as e:
    sys.stderr.write('Failed to read README.md: {}\n'.format(e))
    sys.stderr.flush()
    long_description = ''

# Remove badges since they will always be obsolete.
# This assumes the first 12 lines contain badge info.
long_description = '\n'.join([str(line) for line in long_description.split('\n')[12:]])

# examples/tutorials
EXTRAS_REQUIRE = [
    'jupyter>=1.0.0',
    'graphviz>=0.8',
    'matplotlib>=1.3',
    'torchvision>=0.7.0',
    'visdom>=0.1.4',
    'pandas',
    'scikit-learn',
    'seaborn',
    'wget',
    # 'biopython>=1.54',  # Requires Python 3.6
    # 'scanpy>=1.4',  # Requires HDF5
    # 'scvi>=0.6',  # Requires loopy and other fragile packages
]

setup(
    name='pyro-ppl',
    version=version,
    description='A Python library for probabilistic modeling and inference',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(include=['pyro', 'pyro.*']),
    package_data={"pyro.distributions": ["*.cpp"]},
    author="Uber AI Labs",
    url='http://pyro.ai',
    install_requires=[
        # if you add any additional libraries, please also
        # add them to `docs/requirements.txt`
        # numpy is necessary for some functionality of PyTorch
        'numpy>=1.7',
        'opt_einsum>=2.3.2',
        'pyro-api>=0.1.1',
        'torch>=1.6.0',
        'tqdm>=4.36',
    ],
    extras_require={
        'extras': EXTRAS_REQUIRE,
        'test': EXTRAS_REQUIRE + [
            'nbval',
            'pytest>=5.0',
            'pytest-cov',
            'scipy>=1.1',
        ],
        'profile': ['prettytable', 'pytest-benchmark', 'snakeviz'],
        'dev': EXTRAS_REQUIRE + [
            'flake8',
            'isort',
            'nbformat',
            'nbsphinx>=0.3.2',
            'nbstripout',
            'nbval',
            'ninja',
            'pypandoc',
            'pytest>=5.0',
            'pytest-xdist',
            'scipy>=1.1',
            'sphinx',
            'sphinx_rtd_theme',
            'yapf',
        ],
        'horovod': ['horovod[pytorch]>=0.19'],
        'funsor': [
            # This must be a released version when Pyro is released.
            'funsor[torch]==0.3.0',
        ],
    },
    python_requires='>=3.6',
    keywords='machine learning statistics probabilistic programming bayesian modeling pytorch',
    license='Apache 2.0',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    # yapf
)
