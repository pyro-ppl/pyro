<div align="center">
  <img width="150px" height="150px" src="docs/img/pyro_logo.png">
</div>
-----------------

[![Build Status](https://travis-ci.com/uber/pyro.svg?token=LrMxkQNuTGCmwphBqyVs&branch=dev)](https://travis-ci.com/uber/pyro)

[Examples](examples) | [Getting Started](pyro.ai/tutorial) | [Contributing](CONTRIBUTING.md) | Discussion

Pyro is a flexible, scalable deep probabilistic programming library built on PyTorch.  Notably, it was designed with these principles in mind:
- Universal: Pyro is a universal PPL -- it can represent any computable probability distribution.
- Scalable: Pyro scales to large data sets with little overhead compare to hand-written code.
- Minimal: Pyro is agile and maintainable. It is implemented with a small core of powerful, composable abstractions.
- Flexible: Pyro aims for automation when you want it, control when you need it. This is accomplished through high-level abstractions to express generative and inference models, while allowing experts easy-access to customize inference.

Pyro is developed and maintained by Uber AI Labs.

## Installation

First install [Pytorch](http://pytorch.org/).

Install via pip:
```sh
pip install pyroppl
```

Install from source:
```sh
git clone git@github.com:uber/pyro.git
cd pyro
pip install .
```
