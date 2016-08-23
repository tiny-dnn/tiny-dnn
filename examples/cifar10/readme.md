# Cifar-10 Classification Example

[Cifar-10](http://www.cs.toronto.edu/~kriz/cifar.html) is a common dataset for object classification.
The problem is to classify 32x32 RGB (thus 32x32x3=1024 dimensions) image into 10 classes:
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. 

This problem is more complex than [MNIST](http://yann.lecun.com/exdb/mnist/) classification task.
This means network architecture for Cifar-10 tends to be larger (or/and deeper) than MNIST.
(If you are a machine learning beginner, I recommend you to visit 
[MNIST example](https://github.com/nyanp/tiny-cnn/tree/master/examples/mnist) before this page.)

## Prerequisites for this example
- Download and locate Cifar-10 binary version dataset

## Constructing Model

```cpp
// specify loss-function and learning strategy
network<cross_entropy, adam> nn;

typedef convolutional_layer<activation::identity> conv;
typedef max_pooling_layer<relu> pool;

const int n_fmaps = 32; ///< number of feature maps for upper layer
const int n_fmaps2 = 64; ///< number of feature maps for lower layer
const int n_fc = 64; ///< number of hidden units in fully-connected layer

nn << conv(32, 32, 5, 3, n_fmaps, padding::same)
    << pool(32, 32, n_fmaps, 2)
    << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same)
    << pool(16, 16, n_fmaps, 2)
    << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same)
    << pool(8, 8, n_fmaps2, 2)
    << fully_connected_layer<activation::identity>(4 * 4 * n_fmaps2, n_fc)
    << fully_connected_layer<softmax>(n_fc, 10);

```

## Loading Dataset
```cpp
vector<label_t> train_labels, test_labels;
vector<vec_t> train_images, test_images;

for (int i = 1; i <= 5; i++) {
    parse_cifar10(data_dir_path + "/data_batch_" + to_string(i) + ".bin",
                    &train_images, &train_labels, -1.0, 1.0, 0, 0);
}

parse_cifar10(data_dir_path + "/test_batch.bin",
                &test_images, &test_labels, -1.0, 1.0, 0, 0);
```

# Grid Search
One of the most important hyperparameter in deep learning is learning rate. 
To get stable and better result, let's try [grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search)
for learning rate. The entire code for training cifar-10 is following:

```cpp
#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

template <typename N>
void construct_net(N& nn) {
    typedef convolutional_layer<activation::identity> conv;
    typedef max_pooling_layer<relu> pool;

    const int n_fmaps = 32; ///< number of feature maps for upper layer
    const int n_fmaps2 = 64; ///< number of feature maps for lower layer
    const int n_fc = 64; ///< number of hidden units in fully-connected layer

    nn << conv(32, 32, 5, 3, n_fmaps, padding::same)
        << pool(32, 32, n_fmaps, 2)
        << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same)
        << pool(16, 16, n_fmaps, 2)
        << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same)
        << pool(8, 8, n_fmaps2, 2)
        << fully_connected_layer<activation::identity>(4 * 4 * n_fmaps2, n_fc)
        << fully_connected_layer<softmax>(n_fc, 10);
}

void train_cifar10(string data_dir_path, double learning_rate, ostream& log) {
    // specify loss-function and learning strategy
    network<cross_entropy, adam> nn;

    construct_net(nn);

    log << "learning rate:" << learning_rate << endl;

    cout << "load models..." << endl;

    // load cifar dataset
    vector<label_t> train_labels, test_labels;
    vector<vec_t> train_images, test_images;

    for (int i = 1; i <= 5; i++) {
        parse_cifar10(data_dir_path + "/data_batch_" + to_string(i) + ".bin",
                      &train_images, &train_labels, -1.0, 1.0, 0, 0);
    }

    parse_cifar10(data_dir_path + "/test_batch.bin",
                  &test_images, &test_labels, -1.0, 1.0, 0, 0);

    cout << "start learning" << endl;

    progress_display disp(train_images.size());
    timer t;
    const int n_minibatch = 10; ///< minibatch size
    const int n_train_epochs = 30; ///< training duration

    nn.optimizer().alpha *= sqrt(n_minibatch) * learning_rate;

    // create callback
    auto on_enumerate_epoch = [&]() {
        cout << t.elapsed() << "s elapsed." << endl;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        log << res.num_success << "/" << res.num_total << endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&]() {
        disp += minibatch_size;
    };

    // training
    nn.train(train_images, train_labels, n_minibatch, n_train_epochs, on_enumerate_minibatch, on_enumerate_epoch);

    cout << "end training." << endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(cout);

    // save networks
    ofstream ofs("cifar-weights");
    ofs << nn;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "Usage : " << argv[0]
            << "arg[0]: path_to_data (example:../data)" << endl;
        << "arg[1]: learning rate (example:0.01)" << endl;
        return -1;
    }
    train_cifar10(argv[1], stod(argv[2]), cout);
}
```

compile this file and try various learning rate:

```
./train your-cifar-10-data-directory 10.0
./train your-cifar-10-data-directory 1.0
./train your-cifar-10-data-directory 0.1
./train your-cifar-10-data-directory 0.01
```

>Note:
>If training is too slow, change ```n_training_epochs```, ```n_fmaps``` and ```n_fmaps2``` variables to smaller value.

If you see the following message, some network weights become infinite while training.
Usually it implies too large learning rate.

```
[Warning]Detected infinite value in weight. stop learning.
```

You will get about 70% accuracy in learning rate=0.01.