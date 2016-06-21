# MNIST Reconstruction and Classification with Deconvolutional Auto-encoder

[MNIST](http://yann.lecun.com/exdb/mnist/) is well-known dataset of handwritten digits. We'll use paired convolution and deconvolution with 4 layers in total for an auto-encoder without fully connected layers.

## Prerequisites for this example
- OpenCV

## Intruduction on Implementation Method

We implement the forward pass of deconvolution as a typical upsampling process with convolutional kernels. The backward pass codes are implemented based on the fact that the result of deconvolution is the same as padded convolution with a reversed kernel which means that the element indexs are reversed in view of the result.

## Constructing Model
You can add layers from top to bottom by operator <<, we recommand that the convolution and deconvolution layers be paired as conv1 - conv2...deconv2 - deconv1.

```cpp
    // construct nets
void construct_net(network<sequential>& nn) {
    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6)
       << convolutional_layer<tan_h>(28, 28, 3, 6, 16)
       << deconvolutional_layer<tan_h>(26, 26, 3, 16, 6)
       << deconvolutional_layer<tan_h>(28, 28, 5, 6, 1);
}
```

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

## Saving/Loading models
Just use ```operator <<``` and ```operator >>``` with ostream/istream.

```cpp
std::ofstream ofs("LeNet-weights");
ofs << nn;

std::ifstream ifs("LeNet-weights");
ifs >> nn;
```

## Visulization
This examples will have a visualization of output images and the basic deconvolution kernels, you can modify codes below to have a selection of visualization of layer outputs, convolution kernels and deconvolution kernels.

```cpp
// visualize outputs of each layer
for (size_t i = 0; i < nn.layer_size(); i++) {
    auto out_img = nn[i]->output_to_image();
    cv::imshow("layer:" + std::to_string(i), image2mat(out_img));
}
// visualize filter shape of first convolutional layer
auto weightc = nn.at<convolutional_layer<tan_h>>(0).weight_to_image();
cv::imshow("weights:", image2mat(weightc));
```

You can just doing so:
```
./example_deconv_visual ~/Desktop/4.png ../data/
```
## Classification
We replace a convolutional layer in [LeNet-5](http://yann.lecun.com/exdb/lenet/)-like architecture as a deconvolutional layer for MNIST digits recognition task and got an acceptable accuracy over 98%, you can doing so to carry out the classification example:
```
./example_deconv_train ../data/ deLaNet
```

