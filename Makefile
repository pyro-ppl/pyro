.PHONY: all install docs lint format test integration-test clean FORCE

all: docs test

install: FORCE
	pip install -e .[notebooks,visualization,dev,profile]

docs: FORCE
	$(MAKE) -C docs html

apidoc: FORCE
	$(MAKE) -C docs apidoc

lint: FORCE
	flake8

scrub: FORCE
	find tutorial -name "*.ipynb" | xargs python -m nbstripout --keep-output --keep-count

format: FORCE
	yapf -i *.py pyro/distributions/*.py profiler/*.py docs/source/conf.py
	isort --recursive *.py pyro/ tests/ profiler/*.py docs/source/conf.py

test: lint docs FORCE
	pytest -vx -n auto --stage unit

test-examples: lint FORCE
	pytest -vx -n auto --stage test_examples

test-tutorials: lint FORCE
	CI=1 grep -l smoke_test tutorial/source/*.ipynb \
	  | xargs pytest -vx --nbval-lax

integration-test: lint FORCE
	pytest -vx -n auto --stage integration

test-all: lint FORCE
	pytest -vx -n auto
	CI=1 grep -l smoke_test tutorial/source/*.ipynb \
	  | xargs pytest -vx --nbval-lax

test-cuda: lint FORCE
	PYRO_TENSOR_TYPE=torch.cuda.DoubleTensor pytest -vx -n 8 --stage unit

test-torch-dist: lint FORCE
	PYRO_USE_TORCH_DISTRIBUTIONS=1 pytest -v tests/distributions
	PYRO_USE_TORCH_DISTRIBUTIONS=1 pytest -vx -n auto --stage unit

clean: FORCE
	git clean -dfx -e pyro-egg.info

FORCE:
