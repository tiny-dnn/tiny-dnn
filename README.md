tiny-cnn
========

convolutional neural networks in C++11

build
------
### gcc, without TBB
    ./waf configure
    ./waf build

### vc, with TBB
open vc/tiny_cnn.sln and build in release mode.

### gcc, with TBB
build tiny-cnn with -d CNN_USE_TBB.

dependency
-----
* boost C++ library
* Intel TBB

how to test
-----
1. download MNIST dataset http://yann.lecun.com/exdb/mnist/
2. run program