.PHONY: all install docs lint format test integration-test clean FORCE

all: docs test

install: FORCE
	pip install -e .[notebooks,visualization,dev]

docs: FORCE
	$(MAKE) -C docs html

lint: FORCE
	flake8

format: FORCE
	yapf -i -p *.py pyro/*.py pyro/*/*.py
	isort -i *.py pyro/*.py pyro/*/*.py

test: lint FORCE
	pytest -vx -n auto tests

integration-test: lint FORCE
	pytest -vx -n auto tests --run_integration_tests

clean: FORCE
	git clean -dfx -e pyro-egg.info

FORCE:
