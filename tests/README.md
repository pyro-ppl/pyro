# Testing

### Building PyTorch binaries for Travis CI

The Pyro development on `dev` branch may use the latest features that are not present in the 
release version of PyTorch. As such, these features need to be tested against PyTorch's master 
branch rather than the release branch, which may be a few months older.

For this, we need to build PyTorch binaries on a need-be basis, and upload them to an S3 bucket, 
from which these binaries can be pulled in by CI machines to build Pyro's `dev` branch. Note that 
to build small sized binaries, we remove all CUDA dependencies, but have statically linked MKL
libraries which are not present on CI machines otherwise. 

We use the scripts in the [PyTorch builder](https://github.com/pytorch/builder) repo for this. 
In particular, use the following instructions for building the binaries:
 - Clone the [PyTorch builder](https://github.com/pytorch/builder) repo on your local system.
 - Under the `manywheel` directory, run a docker container:
     ```sh
     cd manywheel
     docker run -it --ipc=host --rm -v $(pwd):/remote soumith/manylinux-cuda80:latest bash
     ```
 - Modify the `build_cpu.sh` script as follows:
   - Remove all environment variables declared in the beginning of the script, except for 
     `NO_CUDA` and `CMAKE_LIBRARY_PATH`.
   - Instead of checking out the version tag through `git checkout tags/v${PYTORCH_BUILD_VERSION}`, 
     check out the commit/branch you would like to build.
 - We need to update `cmake` to version 3 to build PyTorch master:
     ```sh
     yum install epel-release
     yum install cmake3
     yum remove cmake 
     ln -s /usr/bin/cmake3 /usr/bin/cmake
     ```
 - Then run, `./build_cpu.sh` under the `remote` folder, which will create the required binaries 
 in `/wheelhouse` directory.
 - Upload the binaries for python 2.7 (`torch*cp27mu*.whl`) and python 3.5 (`torch*cp35m*.whl`) 
   to the S3 bucket `s3://pyro-ppl/ci` with public read permissions.
 