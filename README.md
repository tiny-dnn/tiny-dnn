tiny-cnn: A C++11 implementation of deep learning (convolutional neural networks)
========

tiny-cnn is a C++11 implementation of deep learning (convolutional neural networks). 

* [designing principles](#designing-principles)
* [comparison with other libraries](#comparison-with-other-libraries)
* [supported networks](#supported-networks)
* [dependencies](#dependencies)
* [building sample project](#building-sample-project)
* [examples](#examples)
* [references](#references)
* [license](#license)

## designing principles
- fast, without GPU
    - with TBB threading and SSE/AVX vectorization
    - 98.8% accuracy on MNIST in 13 minutes training (@Core i7-3520M)
- header only
    - Just include tiny_cnn.h and write your model in c++. There is nothing to install.
- policy-based design
- small dependency & simple implementation

## comparison with other libraries

| |Language|Lines Of Code|License|Prerequisites|Platforms|Modeling By|GPU Support|Installing|Pre-Trained model|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|tiny-cnn|C++|3.1K|BSD(3-clause)|Boost,TBB|Linux/OS-X/Windows|C++ code|No|Unnecessary|No|
|[caffe](https://github.com/BVLC/caffe)|C++(Python/Matlab interfaces available)|58.7K|BSD(2-clause)|CUDA,BLAS,Boost,OpenCV,protobuf,etc|Linux/OS-X|Config File|Yes|Necessary|Yes|
|[Theano](https://github.com/Theano/Theano)|Python|134K|BSD(3-clause)|Numpy,Scipy,BLAS,(optional:nose,Sphinx,CUDA etc)|Linux/OS-X/Windows|Python Code|Yes|Necessary|No|

## supported networks
### layer-types
* fully-connected layer
* fully-connected layer with dropout
* convolutional layer
* average pooling layer
* max-pooling layer

### activation functions
* tanh
* sigmoid
* rectified linear
* identity

### loss functions
* cross-entropy
* mean-squared-error

### optimization algorithm
* stochastic gradient descent (with/without L2 normalization and momentum)
* stochastic gradient levenberg marquardt
* adagrad
* rmsprop

## dependencies
* [boost C++ library](http://www.boost.org/)
* [Intel TBB](https://www.threadingbuildingblocks.org/)

## building sample project
### gcc(4.7~)
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

### vc(2012~)
open vc/tiny_cnn.sln and build in release mode.

You can edit include/config.h to customize default behavior.

## examples
construct convolutional neural networks

```cpp
#include "tiny_cnn.h"
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void construct_cnn() {
    using namespace tiny_cnn;

    // specify loss-function and optimization-algorithm
    network<mse, adagrad> net;
    //network<cross_entropy, RMSprop> net;

    // add layers
    net << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // 32x32in, conv5x5, 1-6 f-maps
        << average_pooling_layer<tan_h>(28, 28, 6, 2) // 28x28in, 6 f-maps, pool2x2
        << fully_connected_layer<tan_h>(14 * 14 * 6, 120)
        << fully_connected_layer<identity>(120, 10);

    assert(net.in_dim() == 32 * 32);
    assert(net.out_dim() == 10);
    
    // load MNIST dataset
    std::vector<label_t> train_labels;
    std::vector<vec_t> train_images;
    
    parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
    parse_mnist_images("train-images.idx3-ubyte", &train_images);
    
    // train (50-epoch, 30-minibatch)
    net.train(train_images, train_labels, 30, 50);
    
    // save
    std::ofstream ofs("weights");
    ofs << net;
    
    // load
    // std::ifstream ifs("weights");
    // ifs >> net;
}
```
construct multi-layer perceptron(mlp)

```cpp
#include "tiny_cnn.h"
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void construct_mlp() {
    network<mse, gradient_descent> net;

    net << fully_connected_layer<sigmoid>(32 * 32, 300);
        << fully_connected_layer<identity>(300, 10);

    assert(net.in_dim() == 32 * 32);
    assert(net.out_dim() == 10);
}
```

another way to construct mlp

```cpp
#include "tiny_cnn.h"
using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void construct_mlp() {
    auto mynet = make_mlp<mse, gradient_descent, tan_h>({ 32 * 32, 300, 10 });

    assert(mynet.in_dim() == 32 * 32);
    assert(mynet.out_dim() == 10);
}
```

more sample, read main.cpp

## references
[1] Y. Bengio, [Practical Recommendations for Gradient-Based Training of Deep Architectures.](http://arxiv.org/pdf/1206.5533v2.pdf) 
    arXiv:1206.5533v2, 2012

[2] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, [Gradient-based learning applied to document recognition.](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
    Proceedings of the IEEE, 86, 2278-2324.
    
other useful reference lists:
- [UFLDL Recommended Readings](http://deeplearning.stanford.edu/wiki/index.php/UFLDL_Recommended_Readings)
- [deeplearning.net reading list](http://deeplearning.net/reading-list/)

## license
The BSD 3-Clause License
