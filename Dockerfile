FROM ubuntu:17.04

MAINTAINER Edgar Riba <edgar.riba@gmail.com>

# Update aptitude with new repo
RUN apt-get update

# Install software
RUN apt-get install -y \
    build-essential    \
    cmake              \
    python-pip         \
    ocl-icd-opencl-dev \ 
    libviennacl-dev \
    git

# Setup software directories
RUN mkdir -p /software

# Setup dependencies
RUN apt-get install -y    \
    libpthread-stubs0-dev \
    libtbb-dev

# Download tiny-dnn
RUN cd /software && \
    git clone https://github.com/tiny-dnn/tiny-dnn.git && \
    cd /software/tiny-dnn && \
    git submodule update --init

# Build tiny-dnn with OpenCL and LibDNN support
RUN cd /software/tiny-dnn && \
    mkdir build && \
    cmake -DBUILD_TESTS=On -DUSE_OPENCL=On -DUSE_LIBDNN=On -Bbuild -H. && \
    cmake --build build
