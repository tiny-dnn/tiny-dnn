# MNIST Digit Classification

[MNIST](http://yann.lecun.com/exdb/mnist/) is well-known dataset of handwritten digits. We'll use [LeNet-5](http://yann.lecun.com/exdb/lenet/)-like architecture for MNIST digits recognition task. LeNet-5 is proposed by Y.LeCun, which is known to work well on handwritten digit recognition. We replace LeNet-5's RBF layer with normal fully-connected layer.

## Prerequisites for this example
- OpenCV

## Constructing Model
Let's define the LeNet network. At first, you have to select loss-function and learning-algorithm by declaration of network class. Then, you can add layers from top to bottom by operator <<.

```cpp
// specify loss-function and learning strategy
network<mse, adagrad> nn;

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
nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6)  // C1, 1@32x32-in, 6@28x28-out
    << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
    << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
            connection_table(tbl, 6, 16))            // C3, 6@14x14-in, 16@10x10-in
    << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
    << convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
    << fully_connected_layer<tan_h>(120, 10);       // F6, 120-in, 10-out
```

What does ```tbl``` mean? LeNet has "sparsity" between S2 and C3 layer. Specifically, each feature map in C3 is connected to a subset of S2's feature maps so that each feature maps get different set of inputs (and hopefully they become compelemtary feature extractor).
tiny-cnn supports this sparsity by ```connection_table``` structure which parameters of constructor are ```bool``` table and number of in/out feature maps.

## Loading Dataset
Tiny-cnn supports idx format, so all you have to do is calling parse_mnist_images and parse_mnist_labels functions.

```cpp
// load MNIST dataset
std::vector<label_t> train_labels, test_labels;
std::vector<vec_t> train_images, test_images;

parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
parse_mnist_images("train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
parse_mnist_labels("t10k-labels.idx1-ubyte", &test_labels);
parse_mnist_images("t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);
```

>Note:
>Original MNIST images are 28x28 centered, [0,255]value.
>This code rescale values [0,255] to [-1.0,1.0], and add 2px borders (so each image is 32x32).

If you want to use another format for learning nets, see [Data Format](https://github.com/nyanp/tiny-cnn/wiki/Data-Format) page.

# Defining Callback
It's convenient if we can check recognition rate on test data, training time, and progress for each epoch while training. Tiny-cnn has callback mechanism for this purpose. We can use local variables(network, test-data, etc) in callback by using C++11's lambda.

```cpp
boost::progress_display disp(train_images.size());
boost::timer t;

// create callbacks
auto on_enumerate_epoch = [&](){
    std::cout << t.elapsed() << "s elapsed." << std::endl;
    tiny_cnn::result res = nn.test(test_images, test_labels);
    std::cout << res.num_success << "/" << res.num_total << std::endl;
    disp.restart(train_images.size());
    t.restart();
};

auto on_enumerate_minibatch = [&](){ disp += minibatch_size; };
```

## Saving/Loading models
Just use ```operator <<``` and ```operator >>``` with ostream/istream.

```cpp
std::ofstream ofs("LeNet-weights");
ofs << nn;

std::ifstream ifs("LeNet-weights");
ifs >> nn;
```

## Putting it all together
train.cpp
```cpp
#include <iostream>
#include "tiny_cnn/tiny_cnn.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;

void construct_net(network<mse, adagrad>& nn) {
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

    // construct nets
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6)  // C1, 1@32x32-in, 6@28x28-out
       << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
            connection_table(tbl, 6, 16))              // C3, 6@14x14-in, 16@10x10-in
       << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
       << fully_connected_layer<tan_h>(120, 10);       // F6, 120-in, 10-out
}

void train_lenet(std::string data_dir_path) {
    // specify loss-function and learning strategy
    network<mse, adagrad> nn;

    construct_net(nn);

    std::cout << "load models..." << std::endl;

    // load MNIST dataset
    std::vector<label_t> train_labels, test_labels;
    std::vector<vec_t> train_images, test_images;

    parse_mnist_labels(data_dir_path+"/train-labels.idx1-ubyte",
                       &train_labels);
    parse_mnist_images(data_dir_path+"/train-images.idx3-ubyte",
                       &train_images, -1.0, 1.0, 2, 2);
    parse_mnist_labels(data_dir_path+"/t10k-labels.idx1-ubyte",
                       &test_labels);
    parse_mnist_images(data_dir_path+"/t10k-images.idx3-ubyte",
                       &test_images, -1.0, 1.0, 2, 2);

    std::cout << "start training" << std::endl;

    progress_display disp(train_images.size());
    timer t;
    int minibatch_size = 10;
    int num_epochs = 30;

    nn.optimizer().alpha *= std::sqrt(minibatch_size);

    // create callback
    auto on_enumerate_epoch = [&](){
        std::cout << t.elapsed() << "s elapsed." << std::endl;
        tiny_cnn::result res = nn.test(test_images, test_labels);
        std::cout << res.num_success << "/" << res.num_total << std::endl;

        disp.restart(train_images.size());
        t.restart();
    };

    auto on_enumerate_minibatch = [&](){
        disp += minibatch_size;
    };

    // training
    nn.train(train_images, train_labels, minibatch_size, num_epochs,
             on_enumerate_minibatch, on_enumerate_epoch);

    std::cout << "end training." << std::endl;

    // test and show results
    nn.test(test_images, test_labels).print_detail(std::cout);

    // save networks
    std::ofstream ofs("LeNet-weights");
    ofs << nn;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage : " << argv[0]
                  << " path_to_data (example:../data)" << std::endl;
        return -1;
    }
    train_lenet(argv[1]);
}
```

>Note:
>Each image has 32x32 values, so dimension of first layer must be equal to 1024.

You'll can get LeNet-Weights binary file after calling sample1_convnet() function. You can also download this file from [here](https://www.dropbox.com/s/mixgjhdi65jm7dl/LeNet-weights?dl=1).

## Use Learned Nets
Here is an example of CUI-based OCR tool.


test.cpp
```cpp
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "tiny_cnn/tiny_cnn.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
    Activation a;
    return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

// convert tiny_cnn::image to cv::Mat and resize
cv::Mat image2mat(image<>& img) {
    cv::Mat ori(img.height(), img.width(), CV_8U, &img.at(0, 0));
    cv::Mat resized;
    cv::resize(ori, resized, cv::Size(), 3, 3, cv::INTER_AREA);
    return resized;
}

void convert_image(const std::string& imagefilename,
    double minv,
    double maxv,
    int w,
    int h,
    vec_t& data) {
    auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
    if (img.data == nullptr) return; // cannot open, or it's not an image

    cv::Mat_<uint8_t> resized;
    cv::resize(img, resized, cv::Size(w, h));

    // mnist dataset is "white on black", so negate required
    std::transform(resized.begin(), resized.end(), std::back_inserter(data),
        [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}


void construct_net(network<mse, adagrad>& nn) {
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

    // construct nets
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6)  // C1, 1@32x32-in, 6@28x28-out
       << average_pooling_layer<tan_h>(28, 28, 6, 2)   // S2, 6@28x28-in, 6@14x14-out
       << convolutional_layer<tan_h>(14, 14, 5, 6, 16,
            connection_table(tbl, 6, 16))              // C3, 6@14x14-in, 16@10x10-in
       << average_pooling_layer<tan_h>(10, 10, 16, 2)  // S4, 16@10x10-in, 16@5x5-out
       << convolutional_layer<tan_h>(5, 5, 5, 16, 120) // C5, 16@5x5-in, 120@1x1-out
       << fully_connected_layer<tan_h>(120, 10);       // F6, 120-in, 10-out
}

void recognize(const std::string& dictionary, const std::string& filename) {
    network<mse, adagrad> nn;

    construct_net(nn);

    // load nets
    ifstream ifs(dictionary.c_str());
    ifs >> nn;

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

    // visualize outputs of each layer
    for (size_t i = 0; i < nn.depth(); i++) {
        auto out_img = nn[i]->output_to_image();
        cv::imshow("layer:" + std::to_string(i), image2mat(out_img));
    }
    // visualize filter shape of first convolutional layer
    auto weight = nn.at<convolutional_layer<tan_h>>(0).weight_to_image();
    cv::imshow("weights:", image2mat(weight));

    cv::waitKey(0);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "please specify image file";
        return 0;
    }
    recognize("LeNet-weights", argv[1]);
}
```

Example image:

![](https://github.com/nyanp/tiny-cnn/wiki/4.bmp)

[https://github.com/nyanp/tiny-cnn/wiki/4.bmp](https://github.com/nyanp/tiny-cnn/wiki/4.bmp)

Compile above code and try to pass 4.bmp, then you can get like:

```
4,95.0808
7,14.9226
9,3.42121
```

This means that the network predict this image as "4", at 95.0808 confidence level.

> Note:
>
> Confidence level may slightly differ on your computer, because of lacking portability of float/double serialization. 

You can also see some images like this:

![](https://github.com/nyanp/tiny-cnn/wiki/weights.bmp)
![](https://github.com/nyanp/tiny-cnn/wiki/layer0.bmp)
![](https://github.com/nyanp/tiny-cnn/wiki/layer1.bmp)
![](https://github.com/nyanp/tiny-cnn/wiki/layer2.bmp)
![](https://github.com/nyanp/tiny-cnn/wiki/layer3.bmp)
![](https://github.com/nyanp/tiny-cnn/wiki/layer4.bmp)
![](https://github.com/nyanp/tiny-cnn/wiki/layer5.bmp)

The first one is learned weight(filter) of first convolutional layer, and others are output values of each layers.
