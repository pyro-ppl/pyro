.PHONY: all install docs lint format test integration-test clean FORCE

all: docs test

install: FORCE
	pip install -e .[dev,profile]

uninstall: FORCE
	pip uninstall pyro-ppl

docs: FORCE
	$(MAKE) -C docs html

apidoc: FORCE
	$(MAKE) -C docs apidoc

tutorial: FORCE
	$(MAKE) -C tutorial html

lint: FORCE
	flake8
	python scripts/update_headers.py --check

license: FORCE
	python scripts/update_headers.py

scrub: FORCE
	find tutorial -name "*.ipynb" | xargs python -m nbstripout --keep-output --keep-count
	find tutorial -name "*.ipynb" | xargs python tutorial/source/cleannb.py

doctest: FORCE
	python -m pytest -p tests.doctest_fixtures --doctest-modules -o filterwarnings=ignore pyro

format: FORCE
	isort --recursive *.py pyro/ examples/ tests/ profiler/*.py docs/source/conf.py

perf-test: FORCE
	bash scripts/perf_test.sh ${ref}

profile: ref=dev

profile: FORCE
	bash scripts/profile_model.sh ${ref} ${models}

test: lint docs doctest FORCE
	pytest -vx -n auto --stage unit

test-examples: lint FORCE
	pytest -vx --stage test_examples

test-tutorials: lint FORCE
	grep -l smoke_test tutorial/source/*.ipynb | xargs grep -L 'smoke_test = False' \
		| CI=1 xargs pytest -vx --nbval-lax --current-env

integration-test: lint FORCE
	pytest -vx -n auto --stage integration

test-all: lint FORCE
	pytest -vx -n auto
	CI=1 grep -l smoke_test tutorial/source/*.ipynb \
	  | xargs pytest -vx --nbval-lax

test-cuda: lint FORCE
	CUDA_TEST=1 PYRO_TENSOR_TYPE=torch.cuda.DoubleTensor pytest -vx --stage unit
	CUDA_TEST=1 pytest -vx tests/test_examples.py::test_cuda

test-cuda-lax: lint FORCE
	CUDA_TEST=1 PYRO_TENSOR_TYPE=torch.cuda.DoubleTensor pytest -vx --stage unit --lax
	CUDA_TEST=1 pytest -vx tests/test_examples.py::test_cuda

test-jit: FORCE
	@echo See jit.log
	pytest -v -n auto --tb=short --runxfail tests/infer/test_jit.py tests/test_examples.py::test_jit | tee jit.log
	pytest -v -n auto --tb=short --runxfail tests/infer/mcmc/test_hmc.py tests/infer/mcmc/test_nuts.py \
		-k JIT=True | tee -a jit.log

clean: FORCE
	git clean -dfx -e pyro_ppl.egg-info

FORCE:
