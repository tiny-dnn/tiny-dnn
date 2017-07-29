FROM ubuntu:17.04

MAINTAINER Edgar Riba <edgar.riba@gmail.com>

# install basic stuff

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential       \
    cmake                 \
    python-pip            \
    python-setuptools     \
    git                   \
    vim

RUN pip install --upgrade pip

# install optional dependencies

RUN apt-get install -y    \
    libpthread-stubs0-dev \
    libtbb-dev

# install linters

RUN apt-get install -y    \
    clang-format-4.0

RUN pip install cpplint
