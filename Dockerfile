FROM ubuntu:16.04

MAINTAINER Edgar Riba <edgar.riba@gmail.com>

# Update aptitude with new repo
RUN apt-get update

# Install software
RUN apt-get install -y \
    build-essential    \
    cmake              \
    python-pip         \
    git

# Setup software directories
RUN mkdir -p /software

# Setup dependencies
RUN apt-get install -y    \
    libpthread-stubs0-dev \
    libtbb-dev

# Download and configure PeachPy
RUN cd /software && \
    git clone https://github.com/Maratyszcza/PeachPy.git && \
    cd /software/PeachPy && \
    pip install -r requirements.txt && \
    python setup.py generate && \
    pip install .

# Download and configure NNPACK
RUN apt-get install ninja-build && \
    pip install ninja-syntax && \
    cd /software && \
    git clone --recursive https://github.com/Maratyszcza/NNPACK.git && \
    cd /software/NNPACK && \
    python ./configure.py && \
    ninja

# Download tiny-dnn
RUN cd /software && \
    git clone https://github.com/tiny-dnn/tiny-dnn.git && \
    cd /software/tiny-dnn && \
    git submodule update --init
