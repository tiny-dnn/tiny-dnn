# Cifar-10 Classification Example

[Cifar-10](http://www.cs.toronto.edu/~kriz/cifar.html) is a common dataset 
for object classification.
The problem is to classify 32x32 RGB (thus 32x32x3=3072 dimensions) image into 10 classes:
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

using conv    = convolutional_layer;
using pool    = max_pooling_layer;
using fc      = fully_connected_layer;
using relu    = relu_layer;
using softmax = softmax_layer;

const int n_fmaps  = 32;  ///< number of feature maps for upper layer
const int n_fmaps2 = 64;  ///< number of feature maps for lower layer
const int n_fc     = 64;  ///< number of hidden units in fully-connected layer

nn << conv(32, 32, 5, 3, n_fmaps, padding::same)          // C1
   << pool(32, 32, n_fmaps, 2)                            // P2
   << relu(16, 16, n_fmaps)                               // activation
   << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same)    // C3
   << pool(16, 16, n_fmaps, 2)                            // P4
   << relu(8, 8, n_fmaps)                                 // activation
   << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same)     // C5
   << pool(8, 8, n_fmaps2, 2)                             // P6
   << relu(4, 4, n_fmaps2)                                // activation
   << fc(4 * 4 * n_fmaps2, n_fc)                          // FC7
   << fc(n_fc, 10) << softmax_layer(10);                  // FC10

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
To get stable and better result, let's try
 [grid search](https://en.wikipedia.org/wiki/Hyperparameter_optimization#Grid_search)
for learning rate. The entire code for training cifar-10 is following:

```cpp
#include <cstdlib>
#include <iostream>
#include <vector>
#include "tiny_dnn/tiny_dnn.h"

#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

template <typename N>
void construct_net(N& nn, core::backend_t backend_type) {
    using conv = convolutional_layer;
    using pool = max_pooling_layer;
    using fc = fully_connected_layer;
    using relu = relu_layer;
    using softmax = softmax_layer;

  const serial_size_t n_fmaps  = 32;  // number of feature maps for upper layer
  const serial_size_t n_fmaps2 = 64;  // number of feature maps for lower layer
  const serial_size_t n_fc     = 64;  // number of hidden units in fc layer

  nn << conv(32, 32, 5, 3, n_fmaps, padding::same, true,
             1, 1, backend_type)                          // C1
     << pool(32, 32, n_fmaps, 2, backend_type)            // P2
     << relu(16, 16, n_fmaps)                             // activation
     << conv(16, 16, 5, n_fmaps, n_fmaps, padding::same,
             true, 1, 1, backend_type)                    // C3
     << pool(16, 16, n_fmaps, 2, backend_type)            // P4
     << relu(8, 8, n_fmaps)                               // activation
     << conv(8, 8, 5, n_fmaps, n_fmaps2, padding::same,
             true, 1, 1, backend_type)                    // C5
     << pool(8, 8, n_fmaps2, 2, backend_type)             // P6
     << relu(4, 4, n_fmaps)                               // activation
     << fc(4 * 4 * n_fmaps2, n_fc, true, backend_type)    // FC7
     << relu(n_fc)                                        // activation
     << fc(n_fc, 10, true, backend_type) << softmax(10);  // FC10
}

void train_cifar10(std::string data_dir_path,
                   double learning_rate,
                   const int n_train_epochs,
                   const int n_minibatch,
                   core::backend_t backend_type,
                   std::ostream &log) {
  // specify loss-function and learning strategy
  network<sequential> nn;
  adam optimizer;

  construct_net(nn, backend_type);

  std::cout << "load models..." << std::endl;

  // load cifar dataset
  std::vector<label_t> train_labels, test_labels;
  std::vector<vec_t> train_images, test_images;

  for (int i = 1; i <= 5; i++) {
    parse_cifar10(data_dir_path + "/data_batch_" + to_string(i) + ".bin",
                  &train_images, &train_labels, -1.0, 1.0, 0, 0);
  }

  parse_cifar10(data_dir_path + "/test_batch.bin", &test_images, &test_labels,
                -1.0, 1.0, 0, 0);

  std::cout << "start learning" << std::endl;

  progress_display disp(train_images.size());
  timer t;

  optimizer.alpha *=
    static_cast<tiny_dnn::float_t>(sqrt(n_minibatch) * learning_rate);

  int epoch = 1;
  // create callback
  auto on_enumerate_epoch = [&]() {
    std::cout << "Epoch " << epoch << "/" << n_train_epochs << " finished. "
              << t.elapsed() << "s elapsed." << std::endl;
    ++epoch;
    tiny_dnn::result res = nn.test(test_images, test_labels);
    log << res.num_success << "/" << res.num_total << std::endl;

    disp.restart(train_images.size());
    t.restart();
  };

  auto on_enumerate_minibatch = [&]() { disp += n_minibatch; };

  // training
  nn.train<cross_entropy>(optimizer, train_images, train_labels, n_minibatch,
                          n_train_epochs, on_enumerate_minibatch,
                          on_enumerate_epoch);

  std::cout << "end training." << std::endl;

  // test and show results
  nn.test(test_images, test_labels).print_detail(std::cout);
  // save networks
  std::ofstream ofs("cifar-weights");
  ofs << nn;
}

static core::backend_t parse_backend_name(const std::string &name) {
  const std::array<const std::string, 5> names = {
    "internal", "nnpack", "libdnn", "avx", "opencl",
  };
  for (size_t i = 0; i < names.size(); ++i) {
    if (name.compare(names[i]) == 0) {
      return static_cast<core::backend_t>(i);
    }
  }
  return core::default_engine();
}

int main(int argc, char **argv) {
  double learning_rate         = 0.1;
  int epochs                   = 30;
  std::string data_path        = "";
  int minibatch_size           = 10;
  core::backend_t backend_type = core::default_engine();
  for (int count = 1; count + 1 < argc; count += 2) {
    std::string argname(argv[count]);
    if (argname == "--learning_rate") {
      learning_rate = atof(argv[count + 1]);
    } else if (argname == "--epochs") {
      epochs = atoi(argv[count + 1]);
    } else if (argname == "--minibatch_size") {
      minibatch_size = atoi(argv[count + 1]);
    } else if (argname == "--backend_type") {
      backend_type = parse_backend_name(argv[count + 1]);
    } else if (argname == "--data_path") {
      data_path = std::string(argv[count + 1]);
    } else if (argname == "--help") {
      std::cout << "Example of usage :\n"
                << argv[0]
                << " --data_path ../data --learning_rate 0.01 --epochs 30 "
                << "--minibatch_size 10 --backend_type internal" << std::endl;
      return 0;
    } else {
      std::cerr << "argument " << argname << " isn't supported. Use --help to "
                << "get usage example";
    }
  }
  if (data_path == "") {
    std::cerr << "Data path not specified. Example of usage :\n"
              << argv[0]
              << " --data_path ../data --learning_rate 0.01 --epochs 30 "
              << "--minibatch_size 10 --backend_type internal" << std::endl;
    return -1;
  }
  if (learning_rate <= 0) {
    std::cerr << "Invalid learning rate. Learning rate must be greater than 0"
              << std::endl;
    return -1;
  }
  if (epochs <= 0) {
    std::cerr << "Invalid epochs number. Epochs number must be greater than 0"
              << std::endl;
    return -1;
  }
  if (minibatch_size <= 0 || minibatch_size > 50000) {
    std::cerr << "Invalid minibatch size. Minibatch rate must be greater than 0"
                 " and less than dataset size (50000)"
              << std::endl;
    return -1;
  }
  std::cout << "Running with following parameters:" << std::endl
            << "Learning rate: " << learning_rate << std::endl
            << "Minibatch size: " << minibatch_size << std::endl
            << "Epochs: " << epochs << std::endl
            << "Backend type: " << backend_type << std::endl;
  train_cifar10(data_path, learning_rate, epochs, minibatch_size, backend_type,
                std::cout);
}
```

compile this file and try various learning rate:

```
./example_cifar_train --data_path your-cifar-10-data-directory --learning_rate 0.01 --epochs 30 --minibatch_size 10 --backend_type internal
./example_cifar_train --data_path your-cifar-10-data-directory --learning_rate 0.01 --epochs 30 --minibatch_size 10 --backend_type internal
```

**Note:** If training is too slow, change ```n_training_epochs```, ```n_fmaps```
 and ```n_fmaps2``` variables to smaller value.

If you see the following message:

```
[Warning]Detected infinite value in weight. stop learning.
```
, some network weights become infinite while training. Usually it implies too 
large learning rate

You will get about 70% accuracy with learning rate=0.01. 
There's a pre-trained weights file named `cifar-weights` in this folder.
