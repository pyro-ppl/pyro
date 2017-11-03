.PHONY: all install docs lint format test integration-test clean FORCE

all: docs test

install: FORCE
	pip install -e .[notebooks,visualization,dev]

docs: FORCE
	$(MAKE) -C docs html

apidoc: FORCE
	$(MAKE) -C docs apidoc

lint: FORCE
	flake8

scrub: FORCE
	find tutorial -name "*.ipynb" | xargs python -m nbstripout --keep-output

format: FORCE
	yapf -i *.py pyro/distributions/*.py docs/source/conf.py
	isort --recursive *.py pyro/ tests/ docs/source/conf.py

test: lint docs FORCE
	pytest -vx -n auto --stage unit

test-examples: lint FORCE
	pytest -vx -n auto --stage test_examples

test-tutorials: lint FORCE
	pytest -v -n auto --nbval-lax tutorial/

integration-test: lint FORCE
	pytest -vx -n auto --stage integration

test-all: lint FORCE
	pytest -vx -n auto

test-cuda: lint FORCE
	PYRO_TENSOR_TYPE=torch.cuda.DoubleTensor pytest -vx -n 8 --stage unit

clean: FORCE
	git clean -dfx -e pyro-egg.info

FORCE:
