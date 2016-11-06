<div align="center">
  <img src="https://github.com/tiny-dnn/tiny-dnn/blob/master/docs/logo/TinyDNN-logo-letters-alpha-version.png"><br><br>
</div>
-----------------

[![Join the chat at https://gitter.im/tiny-dnn/users](https://badges.gitter.im/tiny-dnn/users.svg)](https://gitter.im/tiny-dnn/users) [![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](http://tiny-dnn.readthedocs.io/) [![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://raw.githubusercontent.com/tiny-dnn/tiny-dnn/master/LICENSE) [![Coverage Status](https://coveralls.io/repos/github/tiny-dnn/tiny-dnn/badge.svg?branch=master)](https://coveralls.io/github/tiny-dnn/tiny-dnn?branch=master)

**tiny-dnn** is a C++11 implementation of deep learning. It is suitable for deep learning on limited computational resource, embedded systems and IoT devices.

| **`Linux/Mac OS`** | **`Windows`** |
|------------------|-------------|
|[![Build Status](https://travis-ci.org/tiny-dnn/tiny-dnn.svg?branch=master)](https://travis-ci.org/tiny-dnn/tiny-dnn)|[![Build status](https://ci.appveyor.com/api/projects/status/a5syoifm8ct7b4l2?svg=true)](https://ci.appveyor.com/project/tinydnn/tiny-dnn)|

## Table of contents

* [Features](#features)
* [Comparison with other libraries](#comparison-with-other-libraries)
* [Supported networks](#supported-networks)
* [Dependencies](#dependencies)
* [Build](#build)
* [Examples](#examples)
* [Contributing](#contributing)
* [References](#references)
* [License](#license)
* [Gitter rooms](#gitter-rooms)

Check out the [documentation](http://tiny-dnn.readthedocs.io/) for more info.

## What's New
- 2016/9/14 [tiny-dnn v1.0.0alpha is released!](https://github.com/tiny-dnn/tiny-dnn/releases/tag/v1.0.0a)
- 2016/8/7  tiny-dnn is now moved to organization account, and rename into tiny-dnn :)
- 2016/7/27 [tiny-dnn v0.1.1 released!](https://github.com/tiny-dnn/tiny-dnn/releases/tag/v0.1.1)

## Features
- reasonably fast, without GPU
    - with TBB threading and SSE/AVX vectorization
    - 98.8% accuracy on MNIST in 13 minutes training (@Core i7-3520M)
- portable & header-only
    - Run anywhere as long as you have a compiler which supports C++11
    - Just include tiny_dnn.h and write your model in C++. There is nothing to install.
- easy to integrate with real applications
    - no output to stdout/stderr
    - a constant throughput (simple parallelization model, no garbage collection)
    - work without throwing an exception
    - [can import caffe's model](https://github.com/tiny-dnn/tiny-dnn/tree/master/examples/caffe_converter)
- simply implemented
    - be a good library for learning neural networks

## Comparison with other libraries

||tiny-dnn|[caffe](https://github.com/BVLC/caffe)|[Theano](https://github.com/Theano/Theano)|[TensorFlow](https://www.tensorflow.org/)|
|---|---|---|---|---|
|Prerequisites|__Nothing__(Optional:TBB,OpenMP)|BLAS,Boost,protobuf,glog,gflags,hdf5, (Optional:CUDA,OpenCV,lmdb,leveldb etc)|Numpy,Scipy,BLAS,(optional:nose,Sphinx,CUDA etc)|numpy,six,protobuf,(optional:CUDA,Bazel)|
|Modeling By|C++ code|Config File|Python Code|Python Code|
|GPU Support|No|Yes|Yes|Yes|
|Installing|Unnecessary|Necessary|Necessary|Necessary|
|Windows Support|Yes|No*|Yes|No*|
|Pre-Trained Model|Yes(via caffe-converter)|Yes|No*|No*|

*unofficial version is available

## Supported networks
### layer-types
- core
    - fully-connected
    - dropout
    - linear operation
    - power
- convolution
    - convolutional
    - average pooling
    - max pooling
    - deconvolutional
    - average unpooling
	- max unpooling
- normalization
    - contrast normalization (only forward pass)
    - batch normalization
- split/merge
    - concat
    - slice
    - elementwise-add

### activation functions
* tanh
* sigmoid
* softmax
* rectified linear(relu)
* leaky relu
* identity
* exponential linear units(elu)

### loss functions
* cross-entropy
* mean squared error
* mean absolute error
* mean absolute error with epsilon range

### optimization algorithms
* stochastic gradient descent (with/without L2 normalization and momentum)
* adagrad
* rmsprop
* adam

## Dependencies
Nothing. All you need is a C++11 compiler.

## Build
tiny-dnn is header-ony, so *there's nothing to build*. If you want to execute sample program or unit tests, you need to install [cmake](https://cmake.org/) and type the following commands:

```
cmake .
```

Then open .sln file in visual studio and build(on windows/msvc), or type ```make``` command(on linux/mac/windows-mingw).

Some cmake options are available:

|options|description|default|additional requirements to use|
|-----|-----|----|----|
|USE_TBB|Use [Intel TBB](https://www.threadingbuildingblocks.org/) for parallelization|OFF<sup>1</sup>|[Intel TBB](https://www.threadingbuildingblocks.org/)|
|USE_OMP|Use OpenMP for parallelization|OFF<sup>1</sup>|[OpenMP Compiler](http://openmp.org/wp/openmp-compilers/)|
|USE_SSE|Use Intel SSE instruction set|ON|Intel CPU which supports SSE|
|USE_AVX|Use Intel AVX instruction set|ON|Intel CPU which supports AVX|
|USE_NNPACK|Use NNPACK for convolution operation|OFF|[Acceleration package for neural networks on multi-core CPUs](https://github.com/Maratyszcza/NNPACK)|
|USE_OPENCL|Enable/Disable OpenCL support (experimental)|OFF|[The open standard for parallel programming of heterogeneous systems](https://www.khronos.org/opencl/)|
|USE_LIBDNN|Use Greentea LinDNN for convolution operation with GPU via OpenCL (experimental)|OFF|[An universal convolution implementation supporting CUDA and OpenCL](https://github.com/naibaf7/libdnn)|
|USE_SERIALIZER|Enable model serialization|ON<sup>2</sup>|-|
|BUILD_TESTS|Build unit tests|OFF<sup>3</sup>|-|
|BUILD_EXAMPLES|Build example projects|OFF|-|
|BUILD_DOCS|Build documentation|OFF|[Doxygen](http://www.doxygen.org/)|

<sup>1</sup> tiny-dnn use c++11 standard library for parallelization by default

<sup>2</sup> If you don't use serialization, you can switch off to speedup compilation time.

<sup>3</sup> tiny-dnn uses [Google Test](https://github.com/google/googletest) as default framework to run unit tests. No pre-installation required, it's  automatically downloaded during CMake configuration.

For example, type the following commands if you want to use intel TBB and build tests:
```bash
cmake -DUSE_TBB=ON -DBUILD_TESTS=ON .
```

## Customize configurations
You can edit include/config.h to customize default behavior.

## Examples
construct convolutional neural networks

```cpp
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

void construct_cnn() {
    using namespace tiny_dnn;

    network<sequential> net;

    // add layers
    net << conv<tan_h>(32, 32, 5, 1, 6)  // in:32x32x1, 5x5conv, 6fmaps
        << ave_pool<tan_h>(28, 28, 6, 2) // in:28x28x6, 2x2pooling
        << fc<tan_h>(14 * 14 * 6, 120)   // in:14x14x6, out:120
        << fc<identity>(120, 10);        // in:120,     out:10

    assert(net.in_data_size() == 32 * 32);
    assert(net.out_data_size() == 10);

    // load MNIST dataset
    std::vector<label_t> train_labels;
    std::vector<vec_t> train_images;

    parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);

    // declare optimization algorithm
    adagrad optimizer;

    // train (50-epoch, 30-minibatch)
    net.train<mse>(optimizer, train_images, train_labels, 30, 50);

    // save
    net.save("net");

    // load
    // network<sequential> net2;
    // net2.load("net");
}
```
construct multi-layer perceptron(mlp)

```cpp
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

void construct_mlp() {
    network<sequential> net;

    net << fc<sigmoid>(32 * 32, 300)
        << fc<identity>(300, 10);

    assert(net.in_data_size() == 32 * 32);
    assert(net.out_data_size() == 10);
}
```

another way to construct mlp

```cpp
#include "tiny_dnn/tiny_dnn.h"
using namespace tiny_dnn;
using namespace tiny_dnn::activation;

void construct_mlp() {
    auto mynet = make_mlp<tan_h>({ 32 * 32, 300, 10 });

    assert(mynet.in_data_size() == 32 * 32);
    assert(mynet.out_data_size() == 10);
}
```

more sample, read examples/main.cpp or [MNIST example](https://github.com/tiny-dnn/tiny-dnn/tree/master/examples/mnist) page.

## Contributing
Since deep learning community is rapidly growing, we'd love to get contributions from you to accelerate tiny-dnn development!
For a quick guide to contributing, take a look at the [Contribution Documents](docs/developer_guides/How-to-contribute.md).

## References
[1] Y. Bengio, [Practical Recommendations for Gradient-Based Training of Deep Architectures.](http://arxiv.org/pdf/1206.5533v2.pdf) 
    arXiv:1206.5533v2, 2012

[2] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, [Gradient-based learning applied to document recognition.](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
    Proceedings of the IEEE, 86, 2278-2324.
    
other useful reference lists:
- [UFLDL Recommended Readings](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Recommended_Readings)
- [deeplearning.net reading list](http://deeplearning.net/reading-list/)

## License
The BSD 3-Clause License

## Gitter rooms
We have a gitter rooms for discussing new features & QA.
Feel free to join us!

<table>
<tr>
    <td><b> developers </b></td>
    <td> https://gitter.im/tiny-dnn/developers </td>
</tr>
<tr>
    <td><b> users </b></td>
    <td> https://gitter.im/tiny-dnn/users </td>
</tr>
</table>
