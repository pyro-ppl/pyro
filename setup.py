from setuptools import setup, find_packages

setup(
    name='pyro',
    version='0.0.1',
    description='A Python library for probabilistic modeling and inference',
    packages=find_packages(exclude=('tests*',)),
    install_requires=['numpy>=1.7',
                      'scipy>=0.19.0',
                      'funcy>=1.7.5',
                      'cloudpickle>=0.3.1',
                      'graphviz>=0.8',
                      'networkx>=1.11'
                      ],
    extras_require={
        'notebooks': ['jupyter>=1.0.0'],
        'visualization': ['matplotlib>=1.3',
                          'visdom>=0.1.4']},
    tests_require=['flake8', 'pytest'],
    keywords='machine learning statistics probabilistic programming bayesian modeling pytorch',
    license='MIT License',
    classifiers=['Intended Audience :: Developers',
                 'Intended Audience :: Education',
                 'Intended Audience :: Science/Research',
                 'Operating System :: POSIX :: Linux',
                 'Operating System :: MacOS :: MacOS X',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.4'],
)
