FROM ubuntu:17.04

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

# Upgrade pip
RUN pip install --upgrade pip

# Download and configure PeachPy
RUN pip install --upgrade git+https://github.com/Maratyszcza/PeachPy

# Download and configure confu
RUN pip install --upgrade git+https://github.com/Maratyszcza/confu

# Download and configure NNPACK
RUN apt-get install ninja-build && \
    pip install ninja-syntax && \
    cd /software && \
    git clone --recursive https://github.com/Maratyszcza/NNPACK.git && \
    cd /software/NNPACK && \
    confu setup && \
    python ./configure.py && \
    ninja

# Download tiny-dnn
RUN cd /software && \
    git clone https://github.com/tiny-dnn/tiny-dnn.git && \
    cd /software/tiny-dnn && \
    git submodule update --init
