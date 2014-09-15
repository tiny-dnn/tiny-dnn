tiny-cnn: A C++11 implementation of deep learning (convolutional neural networks)
========

tiny-cnn is a C++11 implementation of deep learning (convolutional neural networks). 

design principle
-----
- fast, without GPU
    - with TBB threading and SSE/AVX vectorization
    - 98.8% accuracy on MNIST in 13 minutes training (@Core i7-3520M)
- header only, policy-based design
- small dependency & simple implementation

supported networks
-----
### layer-types
* fully-connected layer
* fully-connected layer (with dropout)
* convolutional layer
* average pooling layer

### activation functions
* tanh
* sigmoid
* rectified linear
* identity

### loss functions
* cross-entropy
* mean-squared-error

### optimization algorithm
* stochastic gradient descent (with/without L2 normalization)
* stochastic gradient levenberg marquardt

dependencies
-----
* boost C++ library
* Intel TBB

sample code
------

construct convolutional neural networks

```cpp
#include "tiny_cnn.h"
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void cunstruct_cnn() {
    using namespace tiny_cnn;

    // specify loss-function and optimization-algorithm
    typedef network<mse, gradient_descent> CNN;
    CNN mynet;

    // tanh, 32x32 input, 5x5 window, 1-6 feature-maps convolution
    convolutional_layer<CNN, tan_h> C1(32, 32, 5, 1, 6);

    // tanh, 28x28 input, 6 feature-maps, 2x2 subsampling
    average_pooling_layer<CNN, tan_h> S2(28, 28, 6, 2);

    // fully-connected layers
    fully_connected_layer<CNN, sigmoid> F3(14 * 14 * 6, 120);
    fully_connected_layer<CNN, identity> F4(120, 10);

    // connect all
    mynet.add(&C1); mynet.add(&S2); mynet.add(&F3); mynet.add(&F4);

    assert(mynet.in_dim() == 32 * 32);
    assert(mynet.out_dim() == 10);
}
```
construct multi-layer perceptron(mlp)

```cpp
#include "tiny_cnn.h"
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void cunstruct_mlp() {
    typedef network<mse, gradient_descent> MLP;
    MLP mynet;

    fully_connected_layer<MLP, sigmoid> F1(32 * 32, 300);
    fully_connected_layer<MLP, identity> F2(300, 10);

    mynet.add(&F1); mynet.add(&F2);

    assert(mynet.in_dim() == 32 * 32);
    assert(mynet.out_dim() == 10);
}
```

another way to construct mlp

```cpp
#include "tiny_cnn.h"
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void cunstruct_mlp() {
    auto mynet = make_mlp<mse, gradient_descent, tan_h>({ 32 * 32, 300, 10 });

    assert(mynet.in_dim() == 32 * 32);
    assert(mynet.out_dim() == 10);
}
```

more sample, read main.cpp

build sample program
------
### gcc(4.6~)
without tbb

    ./waf configure --BOOST_ROOT=your-boost-root
    ./waf build

with tbb

    ./waf configure --TBB --TBB_ROOT=your-tbb-root --BOOST_ROOT=your-boost-root
    ./waf build

with tbb and SSE/AVX

    ./waf configure --AVX --TBB --TBB_ROOT=your-tbb-root --BOOST_ROOT=your-boost-root
    ./waf build


    ./waf configure --SSE --TBB --TBB_ROOT=your-tbb-root --BOOST_ROOT=your-boost-root
    ./waf build


or edit inlude/config.h to customize default behavior.

### vc(2012~)
open vc/tiny_cnn.sln and build in release mode.

license
------
The BSD 3-Clause License
