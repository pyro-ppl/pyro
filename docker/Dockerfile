ARG base_img=ubuntu:16.04
FROM ${base_img}

# Optional args
ARG cuda=0
ARG python_version=2.7
ARG pyro_branch=release
ARG pytorch_branch=release
ARG uid=1000
ARG gid=1000
ARG ostype=Linux

# Configurable settings
ENV USER_NAME pyromancer
ENV CONDA_DIR /opt/conda
ENV WORK_DIR /workspace
ENV PATH ${CONDA_DIR}/bin:${PATH}

COPY install.sh /install.sh

# Install linux utils
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install conda
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p ${CONDA_DIR} && \
    rm ~/miniconda.sh
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Change to default user
RUN bash -c 'if [ ${ostype} == Linux ]; then groupadd -r --gid ${gid} ${USER_NAME}; fi && \
    useradd -r --create-home --shell /bin/bash --uid ${uid} --gid ${gid} ${USER_NAME}'
WORKDIR ${WORK_DIR}
RUN chown ${USER_NAME} ${WORK_DIR} ${CONDA_DIR} -R
USER ${USER_NAME}

# Install python 2/3; and other packages
RUN conda install -y python=${python_version} && \
    conda install -y jupyter matplotlib

# Install PyTorch and Pyro
RUN bash /install.sh

# Run Jupyter notebook
# (Ref: http://jupyter-notebook.readthedocs.io/en/latest/public_server.html#docker-cmd)
EXPOSE 8888
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
