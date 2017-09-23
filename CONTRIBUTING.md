
# Installing

To install `pyro` for development, we recommend using
[Anaconda or miniconda](https://conda.io/docs/user-guide/install/index.html):

```sh
conda env create -f environment.yml
pip install -e .
```

# Testing

Before submitting a pull request, ensure that linting and unit tests pass locally
```sh
flake8
py.test -v tests/unit
```

To run unit tests locally in parallel, use the `pytest-xdist` package
```sh
pip install pytest-xdist
py.test -v -n auto tests/unit
```
