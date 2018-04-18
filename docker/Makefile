.PHONY: help create-host-workspace build build-gpu run run-gpu notebook notebook-gpu

DOCKER_FILE=Dockerfile
BASE_IMG=ubuntu:16.04
BASE_CUDA_IMG=nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
DOCKER_CMD=docker
DOCKER_GPU_CMD=nvidia-docker
HOST_WORK_DIR=${HOME}/pyro_docker
UID=$(shell id -u)
GID=$(shell id -g)
OSTYPE=$(shell uname)
USER=pyromancer
DOCKER_WORK_DIR=/home/${USER}/workspace/shared

# Optional args
python_version?=2.7
pytorch_branch?=release
pyro_branch?=release
cmd?=bash

# Determine name of docker image
build run notebook: img_prefix=pyro-cpu
build-gpu run-gpu notebook-gpu: img_prefix=pyro-gpu

ifeq ($(img), )
	IMG_NAME=${img_prefix}-${pyro_branch}-${python_version}
else
	IMG_NAME=${img}
endif


help:
	@fgrep -h "##" ${MAKEFILE_LIST} | fgrep -v fgrep | sed -e 's/##//'

##
##Available targets:
##

build: ##
	## Build a docker image for running Pyro on a CPU.
	## Requires nvidia-docker (https://github.com/NVIDIA/nvidia-docker).
	## Args:
	##   python_version: version of python to use. default - python 2.7
	##   pytorch_branch: whether to build PyTorch from conda or from source
	##   (git branch specified by pytorch_branch)
	##     default - latest pytorch version on conda
	##   pyro_branch: whether to use the released Pyro wheel or a git branch.
	##     default - latest pyro-ppl wheel on pypi
	##
	${DOCKER_CMD} build -t ${IMG_NAME} \
	--build-arg base_img=${BASE_IMG} \
	--build-arg uid=${UID} \
	--build-arg gid=${GID} \
	--build-arg ostype=${OSTYPE} \
	--build-arg python_version=${python_version} \
	--build-arg pytorch_branch=${pytorch_branch} \
	--build-arg pyro_branch=${pyro_branch} -f ${DOCKER_FILE} .

build-gpu: ##
	## Build a docker image for running Pyro on a GPU.
	## Requires nvidia-docker (https://github.com/NVIDIA/nvidia-docker).
	## Args:
	##   python_version: version of python to use. default - python 2.7
	##   pytorch_branch: whether to build PyTorch from conda or from source
	##   (git branch specified by pytorch_branch)
	##     default - latest pytorch version on conda
	##   pyro_branch: whether to use the released Pyro wheel or a git branch.
	##     default - latest pyro-ppl wheel on pypi
	##
	${DOCKER_GPU_CMD} build -t ${IMG_NAME} \
	--build-arg base_img=${BASE_CUDA_IMG} \
	--build-arg uid=${UID} \
	--build-arg gid=${GID} \
	--build-arg ostype=${OSTYPE} \
	--build-arg cuda=1 \
	--build-arg python_version=${python_version} \
	--build-arg pytorch_branch=${pytorch_branch} \
	--build-arg pyro_branch=${pyro_branch} -f ${DOCKER_FILE} .

create-host-workspace: ##
	## Create shared volume on the host for sharing files with the container.
	##
	mkdir -p ${HOST_WORK_DIR}

run: create-host-workspace
run: ##
	## Start a Pyro CPU docker instance, and run the command `cmd`.
	## Args:
	##   img: use image name given by `img`.
	##   cmd: command invoked on running a docker instance.
	##     default - bash
	##
	docker run --init -it --user ${USER} \
	-v ${HOST_WORK_DIR}:${DOCKER_WORK_DIR} \
	${IMG_NAME} ${cmd}

run-gpu: create-host-workspace
run-gpu: ##
	## Start a Pyro GPU docker instance, and run the command `cmd`.
	## Args:
	##   img: use image name given by `img`.
	##   cmd: command invoked on running a docker instance.
	##     default - bash
	##
	docker run --init --runtime=nvidia -it --user ${USER} \
	-v ${HOST_WORK_DIR}:${DOCKER_WORK_DIR} \
	${IMG_NAME} ${cmd}

notebook: create-host-workspace
notebook: ##
	## Start a jupyter notebook on the Pyro CPU docker container.
	## Args:
	##   img: use image name given by `img`.
	##
	docker run --init -it -p 8888:8888 --user ${USER} \
	-v ${HOST_WORK_DIR}:${DOCKER_WORK_DIR} \
	${IMG_NAME}

notebook-gpu: create-host-workspace
notebook-gpu: ##
	## Start a juptyer notebook on the Pyro GPU docker container.
	## Args:
	##   img: use image name given by `img`.
	##
	docker run --runtime=nvidia --init -it -p 8888:8888 --user ${USER} \
	-v ${HOST_WORK_DIR}:${DOCKER_WORK_DIR} \
	${IMG_NAME}
