## Using Pyro Docker

Some utilities for building docker images and running Pyro inside a Docker container are
included in the `docker` directory. This includes a Dockerfile to build PyTorch and Pyro,
with some common recipes included in the Makefile.
 
Dependencies for building the docker images:
 - **docker** (>= version 17.05)
 - **nvidia-docker** Refer to the [readme](https://github.com/NVIDIA/nvidia-docker) for
   installation.
 
 
### Building Images

The Makefile can be used to build CPU and CUDA images for Pyro and PyTorch. Some common
options are as follows:

 1. **Source:** Uses the latest released package (conda package for PyTorch and PyPi wheel 
    for Pyro) by default. However, both Pyro and PyTorch can be built from source from the
    master branch or any other arbitrary branch specified by `pytorch_branch` and 
    `pyro_branch`.
 2. **CPU / CUDA:** `make build` or `make build-gpu` can be used to specify whether the CPU
    or the CUDA image is to be built. For building the CUDA image, *nvidia-docker* is 
    required. 
 3. **Python Version:** Python version can be specified via the argument `python_version`. 
 
For example, the `make` command to build an image that uses Pyro's `dev` branch over
PyTorch's `master` branch, using python 3.6 to run on a GPU, is as follows:

```sh
make build-gpu pyro_branch=dev pytorch_branch=master python_version=3.6
```  

This will build an image named `pyro-gpu-dev-3.6`. To spin up a docker container from this
image, and run jupyter notebook on this, use the following `make` command:

```sh
make notebook-gpu img=pyro-gpu-dev-3.6
```

For help on the `make` commands available, run `make help`.

**NOTE (Mac Users)**: Please increase the memory available to the Docker application
via *Preferences --> Advanced* from 2GB (default) to at least 4GB prior to building the
docker image (specially for building PyTorch from source).

### Running the Docker container

Once the image is built, the docker container can be started via `make run`, or 
`make run-gpu`. By default this starts a *bash* shell. One could start an *ipython* 
shell instead by running `make run cmd=ipython`. The image to be used can be 
specified via the argument `img`. 

To run a *jupyter notebook* use `make notebook`, or `make notebook-gpu`. This will 
start a jupyter notebook server which can be accessed from the browser using the link 
mentioned in the terminal. 

Note that there is a shared volume between the container and the host system, with the 
location `$DOCKER_WORK_DIR` on the container, and `$HOST_WORK_DIR` on the local system.
These variables can be configured in the `Makefile`.
