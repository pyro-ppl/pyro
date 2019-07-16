from __future__ import absolute_import, division, print_function

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

# Convert README.md to rst for display at https://pypi.python.org/pypi/pyro-ppl
# When releasing on pypi, make sure pandoc is on your system:
# $ brew install pandoc          # OS X
# $ sudo apt-get install pandoc  # Ubuntu Linux
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
    print(long_description)
except Exception as e:
    sys.stderr.write('Failed to convert README.md to rst:\n  {}\n'.format(e))
    sys.stderr.flush()
    long_description = open('README.md').read()

# Remove badges since they will always be obsolete.
# This assumes the first 10 lines contain badge info.
long_description = '\n'.join([str(line) for line in long_description.split('\n')[10:]])

# examples/tutorials
EXTRAS_REQUIRE = [
    'jupyter>=1.0.0',
    'matplotlib>=1.3',
    'pillow',
    'torchvision>=0.2.2',
    'visdom>=0.1.4',
    'pandas',
    'seaborn',
    'wget',
]

if sys.version_info[0] == 2:
    EXTRAS_REQUIRE.append('functools32')

setup(
    name='pyro-ppl',
    version=version,
    description='A Python library for probabilistic modeling and inference',
    long_description=long_description,
    packages=find_packages(include=['pyro', 'pyro.*']),
    package_data={"pyro.distributions": ["*.cpp"]},
    url='http://pyro.ai',
    author='Uber AI Labs',
    author_email='pyro@uber.com',
    install_requires=[
        # if you add any additional libraries, please also
        # add them to `docs/requirements.txt`
        'contextlib2',
        'graphviz>=0.8',
        # numpy is necessary for some functionality of PyTorch
        'numpy>=1.7',
        'opt_einsum>=2.3.2',
        'six>=1.10.0',
        'torch>=1.1.0',
        'tqdm>=4.31',
    ],
    extras_require={
        'extras': EXTRAS_REQUIRE,
        'test': EXTRAS_REQUIRE + [
            'nbval',
            'pytest>=4.1',
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
            'pytest>=4.1',
            'pytest-xdist',
            'scipy>=1.1',
            'sphinx',
            'sphinx_rtd_theme',
            'yapf',
        ],
    },
    tests_require=['flake8', 'pytest>=4.1'],
    keywords='machine learning statistics probabilistic programming bayesian modeling pytorch',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ],
    # yapf
)
