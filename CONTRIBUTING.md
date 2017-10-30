# Development
For consistency, please adhere to the established coding style including variable names, module imports, and function definitions.  The Pyro codebase uses the PEP8 style guide.

# Setup

Install all the dev dependencies for Pyro.

```sh
pip install -e .[dev]
```

Add the repository's git configuration to your local project `.gitconfig` file.

```sh
git config --local include.path ../.gitconfig
```

# Testing

Before submitting a pull request, please ensure that linting and unit tests pass locally
```sh
flake8
make test
```
Use the `--run_integration_tests` option to run integration tests, but be aware that these may take a long time (hours) to complete. 

To run unit tests locally in parallel, use the `pytest-xdist` package
```sh
pip install pytest-xdist
py.test -vs -n auto
```

To run a single test from the command line
```sh
py.test -vs {path_to_test}::{test_name}
```

# Submitting
For larger changes, please open an issue for discussion before submitting a pull request.  In your PR, please include:
- Changes made
- Links to related issues/PRs
- Tests
- Dependencies

Please add the 'awaiting review' tag and add any of the core Pyro contributors as reviewers.
