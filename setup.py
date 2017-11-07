from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup

setup(
    name='pyro-ppl',
    version='0.1.1',
    description='A Python library for probabilistic modeling and inference',
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
            'pytest-xdist',
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
