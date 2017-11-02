# Development

Please follow our established coding style including variable names, module imports, and function definitions.
The Pyro codebase follows the [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/)
(which you can check with `make lint`) and follows
[`isort`](https://github.com/timothycrosley/isort) import order (which you can enforce with `make format`).

# Setup

First install [PyTorch](http://pytorch.org/).

Then, install all the dev dependencies for Pyro.
```sh
make install
```
or explicitly
```sh
pip install -e .[dev]
```

# Testing

Before submitting a pull request, please autoformat code and ensure that unit tests pass locally
```sh
make format            # runs isort
make test              # linting and unit tests
```

If you've modified core pyro code, examples, or tutorials, you can run more comprehensive tests locally
```sh
make test-examples     # test examples/
make integration-test  # longer-running tests (may take hours)
make test-cuda         # runs unit tests in cuda mode
```

To run all tests locally in parallel, use the `pytest-xdist` package
```sh
pip install pytest-xdist
pytest -vs -n auto
```

To run a single test from the command line
```sh
pytest -vs {path_to_test}::{test_name}
```

# Submitting

For larger changes, please open an issue for discussion before submitting a pull request.
In your PR, please include:
- Changes made
- Links to related issues/PRs
- Tests
- Dependencies

Please add the `awaiting review` tag and add any of the core Pyro contributors as reviewers.
For speculative changes meant for early-stage review, add the `WIP` tag.
Our policy is to require the reviewer to merge the PR, rather than the PR author.
