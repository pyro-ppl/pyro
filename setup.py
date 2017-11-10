from __future__ import absolute_import, division, print_function

import sys

from setuptools import find_packages, setup

# Find pyro version.
for line in open('pyro/__init__.py'):
    if line.startswith('__version__ = '):
        version = line.strip().split()[2][1:-1]

# Convert README.md to rst for display at https://pypi.python.org/pypi/pyro-ppl
# When releasing on pypi, make sure pandoc is on your system:
# $ brew install pandoc          # OS X
# $ sudo apt-get install pandoc  # Ubuntu Linux
try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError, OSError) as e:
    sys.stderr.write('Failed to convert README.md to rst:\n  {}\n'.format(e))
    sys.stderr.flush()
    long_description = open('README.md').read()

# Remove badges since they will always be obsolete.
blacklist = ['Build Status', 'Latest Version', 'travis-ci.org', 'pypi.python.org']
long_description = '\n'.join([
    line for line in long_description.split('\n')
    if not any(patt in line for patt in blacklist)
])

setup(
    name='pyro-ppl',
    version=version,
    description='A Python library for probabilistic modeling and inference',
    long_description=long_description,
    packages=find_packages(exclude=('tests*',)),
    url='http://pyro.ai',
    author='Uber AI Labs',
    author_email='pyro@uber.com',
    install_requires=[
        'numpy>=1.7',
        'scipy>=0.19.0',
        'cloudpickle>=0.3.1',
        'graphviz>=0.8',
        'networkx>=2.0.0',
        'observations>=0.1.4',
        'torch',
        'six>=1.10.0',
    ],
    extras_require={
        'notebooks': ['jupyter>=1.0.0'],
        'visualization': [
            'matplotlib>=1.3',
            'visdom>=0.1.4',
            'pillow',
        ],
        'test': [
            'pytest',
            'pytest-cov',
            'nbval',
            # examples/tutorials
            'visdom',
            'torchvision',
        ],
        'dev': [
            'torchvision',
            'flake8',
            'yapf',
            'isort',
            'pytest',
            'pytest-xdist',
            'nbval',
            'nbstripout',
            'pypandoc',
            'sphinx',
            'sphinx_rtd_theme',
        ],
    },
    tests_require=['flake8', 'pytest'],
    keywords='machine learning statistics probabilistic programming bayesian modeling pytorch',
    license='MIT License',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
    ],
    # yapf
)
