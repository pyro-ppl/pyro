# Testing

Before submitting a pull request, ensure that linting and unit tests pass locally
```sh
flake8
py.test tests/unit -v
```

To run unit tests locally in parallel, use the `pytest-xdist` package
```sh
pip install pytest-xdist
py.test -n auto tests/unit
```
