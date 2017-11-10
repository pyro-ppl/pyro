<div align="center">
  <a href="http://pyro.ai"> <img width="150px" height="150px" src="docs/source/_static/img/pyro_logo.png"></a>
</div>


----------------------------------------------------------------

[![Build Status](https://travis-ci.org/uber/pyro.svg?branch=master)](https://travis-ci.org/uber/pyro)
[![Latest Version](https://badge.fury.io/py/pyro-ppl.svg)](https://pypi.python.org/pypi/pyro-ppl)


[Getting Started](http://pyro.ai/examples) |
[Documentation](http://docs.pyro.ai/) |
[Contributing](https://github.com/uber/pyro/blob/master/CONTRIBUTING.md)

Please also refer to the [Pyro homepage](http://pyro.ai/).

Pyro is a flexible, scalable deep probabilistic programming library built on PyTorch.  Notably, it was designed with these principles in mind:
- **Universal**: Pyro is a universal PPL -- it can represent any computable probability distribution.
- **Scalable**: Pyro scales to large data sets with little overhead compared to hand-written code.
- **Minimal**: Pyro is agile and maintainable. It is implemented with a small core of powerful, composable abstractions.
- **Flexible**: Pyro aims for automation when you want it, control when you need it. This is accomplished through high-level abstractions to express generative and inference models, while allowing experts easy-access to customize inference.

Pyro is in an alpha release.  It is developed and used by [Uber AI Labs](http://uber.ai).
More information is available in the [launch blog post](http://eng.uber.com/pyro).

## Installation

First install [PyTorch](http://pytorch.org/).

Install via pip:

**Python 2.7.\*:**
```sh
pip install pyro-ppl
```

**Python 3.5:**
```
pip3 install pyro-ppl
```

**Install from source:**
```sh
git clone git@github.com:uber/pyro.git
cd pyro
pip install .
```
