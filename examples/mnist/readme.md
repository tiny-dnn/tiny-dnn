# MNIST Digit Classification

[MNIST](http://yann.lecun.com/exdb/mnist/) is a well-known dataset of handwritten digits. We'll use [LeNet-5](http://yann.lecun.com/exdb/lenet/)-like architecture for MNIST digit recognition task. LeNet-5 is proposed by Y.LeCun, which is known to work well on handwritten digit recognition. We replace LeNet-5's RBF layer with normal fully-connected layer.

## Constructing Model
Let's define the LeNet network. At first, you have to specify loss-function and learning-algorithm. Then, you can add layers from top to bottom by operator <<.

```cpp
// specify loss-function and learning strategy
network<sequential> nn;
adagrad optimizer;

// connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
static const bool tbl [] = {
    O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
    O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
    O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
    X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
    X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
    X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X

// construct nets
//
// C : convolution
// S : sub-sampling
// F : fully connected
nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6,  // C1, 1@32x32-in, 6@28x28-out
        padding::valid, true, 1, 1, backend_type)
   << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
   << convolutional_layer<tan_h>(14, 14, 5, 6, 16, // C3, 6@14x14-in, 16@10x10-in
        connection_table(tbl, 6, 16),
        padding::valid, true, 1, 1, backend_type)
   << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
   << convolutional_layer<tan_h>(5, 5, 5, 16, 120, // C5, 16@5x5-in, 120@1x1-out
        padding::valid, true, 1, 1, backend_type)
   << fully_connected_layer<tan_h>(120, 10,        // F6, 120-in, 10-out
        true, backend_type)
```

What does ```tbl``` mean? LeNet has "sparsity" between S2 and C3 layer. Specifically, each feature map in C3 is connected to a subset of S2's feature maps so that each of the feature maps gets different set of inputs (and hopefully they become compelemtary feature extractors).
Tiny-dnn supports this sparsity by ```connection_table``` structure which parameters of constructor are ```bool``` table and number of in/out feature maps.

## Loading Dataset
Tiny-dnn supports MNIST idx format, so all you have to do is calling parse_mnist_images and parse_mnist_labels functions.

```cpp
// load MNIST dataset
std::vector<label_t> train_labels, test_labels;
std::vector<vec_t> train_images, test_images;

parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
                   &train_labels);
parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
                   &train_images, -1.0, 1.0, 2, 2);
parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                   &test_labels);
parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                   &test_images, -1.0, 1.0, 2, 2);
```

>Note:
>Original MNIST images are 28x28 centered, [0,255]value.
>This code rescale values [0,255] to [-1.0,1.0], and add 2px borders (so each image is 32x32).

If you want to use another format for learning nets, see [Data Format](https://github.com/tiny-dnn/tiny-dnn/wiki/Data-Format) page.

# Defining Callback
It's convenient if we can check recognition rate on test data, training time, and progress for each epoch while training. Tiny-dnn has callback mechanism for this purpose. We can use local variables(network, test-data, etc) in callback by using C++11's lambda.

```cpp
progress_display disp(static_cast<unsigned long>(train_images.size()));
timer t;
int minibatch_size = 10;
int num_epochs = 30;

optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));

// create callback
auto on_enumerate_epoch = [&](){
    std::cout << t.elapsed() << "s elapsed." << std::endl;
    tiny_dnn::result res = nn.test(test_images, test_labels);
    std::cout << res.num_success << "/" << res.num_total << std::endl;

    disp.restart(static_cast<unsigned long>(train_images.size()));
    t.restart();
};

auto on_enumerate_minibatch = [&](){
    disp += minibatch_size;
};
```

## Saving/Loading models
Just use ```network::save(filename)``` and ```network.load(filename)``` to write your whole model as binary file.

```cpp
nn.save("LeNet-model");
nn.load("LeNet-model");
```

## Putting it all together
train.cpp
```cpp
#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;

static void construct_net(network<sequential>& nn) {
    // connection table [Y.Lecun, 1998 Table.1]
#define O true
#define X false
    static const bool tbl[] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

    // by default will use backend_t::tiny_dnn unless you compiled
    // with -DUSE_AVX=ON and your device supports AVX intrinsics
    core::backend_t backend_type = core::default_engine();

    // construct nets
    //
    // C : convolution
    // S : sub-sampling
    // F : fully connected
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6,  // C1, 1@32x32-in, 6@28x28-out
            padding::valid, true, 1, 1, backend_type)
       << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16, // C3, 6@14x14-in, 16@10x10-in
            connection_table(tbl, 6, 16),
            padding::valid, true, 1, 1, backend_type)
       << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120, // C5, 16@5x5-in, 120@1x1-out
            padding::valid, true, 1, 1, backend_type)
       << fully_connected_layer<tan_h>(120, 10,        // F6, 120-in, 10-out
            true, backend_type)
    ;
}

static void train_lenet(const std::string& data_dir_path) {
    // specify loss-function and learning strategy
    network<sequential> nn;
    adagrad optimizer;

    construct_net(nn);

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels(data_dir_path + "/train-labels.idx1-ubyte",
                       &train_labels);
    parse_mnist_images(data_dir_path + "/train-images.idx3-ubyte",
                       &train_images, -1.0, 1.0, 2, 2);
    parse_mnist_labels(data_dir_path + "/t10k-labels.idx1-ubyte",
                       &test_labels);
    parse_mnist_images(data_dir_path + "/t10k-images.idx3-ubyte",
                       &test_images, -1.0, 1.0, 2, 2);

    std::cout << "start training" << std::endl;

    progress_display disp(static_cast<unsigned long>(train_images.size()));
    timer t;
    int minibatch_size = 10;
    int num_epochs = 30;

    optimizer.alpha *= static_cast<tiny_dnn::float_t>(std::sqrt(minibatch_size));

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_dnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(static_cast<unsigned long>(train_images.size()));
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
    };

    // training
    nn.train<mse>(optimizer, train_images, train_labels, minibatch_size, num_epochs,
             on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save network model & trained weights
    nn.save("LeNet-model");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage : " << argv[0]
                  << " path_to_data (example:../data)" << std::endl;
        return -1;
    }
    train_lenet(argv[1]);
    return 0;
}
```

>Note:
>Each image has 32x32 values, so dimension of first layer must be equal to 1024.

You'll be able to get LeNet-model binary file after calling train_lenet() function. You can also download this file from [here](https://www.dropbox.com/s/mixgjhdi65jm7dl/LeNet-weights?dl=1).

## Use Learned Nets
Here is an example of CUI-based OCR tool.


test.cpp
```cpp
#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace std;

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
    Activation a;
    return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

void convert_image(const std::string& imagefilename,
    double minv,
    double maxv,
    int w,
    int h,
    vec_t& data) {

    image<> img(imagefilename, image_type::grayscale);
    image<> resized = resize_image(img, w, h);

    // mnist dataset is "white on black", so negate required
    std::transform(resized.begin(), resized.end(), std::back_inserter(data),
        [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

void recognize(const std::string& dictionary, const std::string& filename) {
    network<sequential> nn;

    nn.load(dictionary);

    // convert imagefile to vec_t
    vec_t data;
    convert_image(filename, -1.0, 1.0, 32, 32, data);

    // recognize
    auto res = nn.predict(data);
    vector<pair<double, int> > scores;

    // sort & print top-3
    for (int i = 0; i < 10; i++)
        scores.emplace_back(rescale<tan_h>(res[i]), i);

    sort(scores.begin(), scores.end(), greater<pair<double, int>>());

    for (int i = 0; i < 3; i++)
        cout << scores[i].second << "," << scores[i].first << endl;

    // save outputs of each layer
    for (size_t i = 0; i < nn.depth(); i++) {
        auto out_img = nn[i]->output_to_image();
        auto filename = "layer_" + std::to_string(i) + ".png";
        out_img.save(filename);
    }
    // save filter shape of first convolutional layer
    {
        auto weight = nn.at<convolutional_layer<tan_h>>(0).weight_to_image();
        auto filename = "weights.png";
        weight.save(filename);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "please specify image file";
        return 0;
    }
    recognize("LeNet-model", argv[1]);
}
```

Example image:

![](https://github.com/tiny-dnn/tiny-dnn/wiki/4.bmp)

[https://github.com/tiny-dnn/tiny-dnn/wiki/4.bmp](https://github.com/tiny-dnn/tiny-dnn/wiki/4.bmp)

Compile above code and try to pass 4.bmp, then you can get like:

```
4,78.1403
7,33.5718
8,14.0017
```

This means that the network predicted this image as "4", at confidence level of 78.1403%.

> Note:
>
> Confidence level may slightly differ on your computer, stay tuned! 

You can also see some images like this:

![](https://github.com/tiny-dnn/tiny-dnn/wiki/weights.bmp)
![](https://github.com/tiny-dnn/tiny-dnn/wiki/layer0.bmp)
![](https://github.com/tiny-dnn/tiny-dnn/wiki/layer1.bmp)
![](https://github.com/tiny-dnn/tiny-dnn/wiki/layer2.bmp)
![](https://github.com/tiny-dnn/tiny-dnn/wiki/layer3.bmp)
![](https://github.com/tiny-dnn/tiny-dnn/wiki/layer4.bmp)
![](https://github.com/tiny-dnn/tiny-dnn/wiki/layer5.bmp)

The first one is learned weights(filter) of first convolutional layer, and others are output values of each of the layers.
