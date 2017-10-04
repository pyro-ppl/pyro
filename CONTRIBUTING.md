# Testing

Before submitting a pull request, ensure that linting and unit tests pass locally
```sh
flake8
py.test -vs
```
Use the `--run_integration_tests` option to include running integration tests, but be aware that these may take a long time (hours) to complete. 

To run unit tests locally in parallel, use the `pytest-xdist` package
```sh
pip install pytest-xdist
py.test -vs -n auto
```

To run a single test from the command line
```sh
py.test -vs {path_to_test}::{test_name}
```
