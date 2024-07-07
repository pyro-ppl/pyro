<!--
Copyright Contributors to the Pyro project.

SPDX-License-Identifier: Apache-2.0
-->

<div align="center">
  <a href="http://pyro.ai"> <img width="220px" height="220px" src="docs/source/_static/img/pyro_logo_with_text.png"></a>
</div>

-----------------------------------------

[![Build Status](https://github.com/pyro-ppl/pyro/workflows/CI/badge.svg)](https://github.com/pyro-ppl/pyro/actions)
[![Coverage Status](https://coveralls.io/repos/github/pyro-ppl/pyro/badge.svg?branch=dev)](https://coveralls.io/github/pyro-ppl/pyro?branch=dev)
[![Latest Version](https://badge.fury.io/py/pyro-ppl.svg)](https://pypi.python.org/pypi/pyro-ppl)
[![Documentation Status](https://readthedocs.org/projects/pyro-ppl/badge/?version=dev)](http://pyro-ppl.readthedocs.io/en/stable/?badge=dev)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/3056/badge)](https://bestpractices.coreinfrastructure.org/projects/3056)

[Getting Started](http://pyro.ai/examples) |
[Documentation](http://docs.pyro.ai/) |
[Community](http://forum.pyro.ai/) |
[Contributing](https://github.com/pyro-ppl/pyro/blob/master/CONTRIBUTING.md)

Pyro is a flexible, scalable deep probabilistic programming library built on PyTorch.  Notably, it was designed with these principles in mind:

- **Universal**: Pyro is a universal PPL - it can represent any computable probability distribution.
- **Scalable**: Pyro scales to large data sets with little overhead compared to hand-written code.
- **Minimal**: Pyro is agile and maintainable. It is implemented with a small core of powerful, composable abstractions.
- **Flexible**: Pyro aims for automation when you want it, control when you need it. This is accomplished through high-level abstractions to express generative and inference models, while allowing experts easy-access to customize inference.

Pyro was originally developed at Uber AI and is now actively maintained by community contributors, including a dedicated team at the [Broad Institute](https://www.broadinstitute.org/).
In 2019, Pyro [became](https://www.linuxfoundation.org/press-release/2019/02/pyro-probabilistic-programming-language-becomes-newest-lf-deep-learning-project/) a project of the Linux Foundation, a neutral space for collaboration on open source software, open standards, open data, and open hardware.

For more information about the high level motivation for Pyro, check out our [launch blog post](http://eng.uber.com/pyro).
For additional blog posts, check out work on [experimental design](https://eng.uber.com/oed-pyro-release/) and
[time-to-event modeling](https://eng.uber.com/modeling-censored-time-to-event-data-using-pyro/) in Pyro.

## Installing

### Installing a stable Pyro release

**Install using pip:**
```sh
pip install pyro-ppl
```

**Install from source:**
```sh
git clone git@github.com:pyro-ppl/pyro.git
cd pyro
git checkout master  # master is pinned to the latest release
pip install .
```

**Install with extra packages:**

To install the dependencies required to run the probabilistic models included in the `examples`/`tutorials` directories, please use the following command:
```sh
pip install pyro-ppl[extras] 
```
Make sure that the models come from the same release version of the [Pyro source code](https://github.com/pyro-ppl/pyro/releases) as you have installed.

### Installing Pyro dev branch

For recent features you can install Pyro from source.

**Install Pyro using pip:**

```sh
pip install git+https://github.com/pyro-ppl/pyro.git
```
or, with the `extras` dependency to run the probabilistic models included in the `examples`/`tutorials` directories:
```sh
pip install git+https://github.com/pyro-ppl/pyro.git#egg=project[extras]
```

**Install Pyro from source:**

```sh
git clone https://github.com/pyro-ppl/pyro
cd pyro
pip install .  # pip install .[extras] for running models in examples/tutorials
```

## Running Pyro from a Docker Container

Refer to the instructions [here](docker/README.md).

## Citation
If you use Pyro, please consider citing:
```
@article{bingham2019pyro,
  author    = {Eli Bingham and
               Jonathan P. Chen and
               Martin Jankowiak and
               Fritz Obermeyer and
               Neeraj Pradhan and
               Theofanis Karaletsos and
               Rohit Singh and
               Paul A. Szerlip and
               Paul Horsfall and
               Noah D. Goodman},
  title     = {Pyro: Deep Universal Probabilistic Programming},
  journal   = {J. Mach. Learn. Res.},
  volume    = {20},
  pages     = {28:1--28:6},
  year      = {2019},
  url       = {http://jmlr.org/papers/v20/18-403.html}
}
```
