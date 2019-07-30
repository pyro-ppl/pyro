<div align="center">
  <a href="http://pyro.ai"> <img width="220px" height="220px" src="docs/source/_static/img/pyro_logo_with_text.png"></a>
</div>

-----------------------------------------

[![Build Status](https://travis-ci.com/pyro-ppl/pyro.svg?branch=dev)](https://travis-ci.com/uber/pyro)
[![codecov.io](https://codecov.io/github/pyro-ppl/pyro/branch/dev/graph/badge.svg)](https://codecov.io/github/uber/pyro)
[![Latest Version](https://badge.fury.io/py/pyro-ppl.svg)](https://pypi.python.org/pypi/pyro-ppl)
[![Documentation Status](https://readthedocs.org/projects/pyro-ppl/badge/?version=dev)](http://pyro-ppl.readthedocs.io/en/stable/?badge=dev)

[Getting Started](http://pyro.ai/examples) |
[Documentation](http://docs.pyro.ai/) |
[Community](http://forum.pyro.ai/) |
[Contributing](https://github.com/pyro-ppl/pyro/blob/master/CONTRIBUTING.md)

Pyro is a flexible, scalable deep probabilistic programming library built on PyTorch.  Notably, it was designed with these principles in mind:

- **Universal**: Pyro is a universal PPL - it can represent any computable probability distribution.
- **Scalable**: Pyro scales to large data sets with little overhead compared to hand-written code.
- **Minimal**: Pyro is agile and maintainable. It is implemented with a small core of powerful, composable abstractions.
- **Flexible**: Pyro aims for automation when you want it, control when you need it. This is accomplished through high-level abstractions to express generative and inference models, while allowing experts easy-access to customize inference.

Pyro is in a beta release.  It is developed and maintained by [Uber AI Labs](http://uber.ai) and community contributors.
For more information, check out our [blog post](http://eng.uber.com/pyro).

## Installing

### Installing a stable Pyro release

**Install using pip:**

Pyro supports Python 3.4+.

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

**Install using pip:**

```sh
pip install git+https://github.com/pyro-ppl/pyro.git
```

or, with the `extras` dependency to run the probabilistic models included in the `examples`/`tutorials` directories:
```sh
pip install git+https://github.com/pyro-ppl/pyro.git#egg=project[extras]
```

**Install from source:**

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
@article{bingham2018pyro,
  author = {Bingham, Eli and Chen, Jonathan P. and Jankowiak, Martin and Obermeyer, Fritz and
            Pradhan, Neeraj and Karaletsos, Theofanis and Singh, Rohit and Szerlip, Paul and
            Horsfall, Paul and Goodman, Noah D.},
  title = {{Pyro: Deep Universal Probabilistic Programming}},
  journal = {arXiv preprint arXiv:1810.09538},
  year = {2018}
}
```
