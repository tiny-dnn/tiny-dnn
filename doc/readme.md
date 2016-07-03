# tiny-cnn documentations

tiny-cnn is a header only, dependency free deep learning library written in C++. It is designed to be used in the real applications, including IoT devices and embedded systems.

## User guides
- [How-to](How-Tos.md)
- [Train network with your dataset](Train-network-with-your-dataset.md)
- [Integrate with your application](Integrate-with-your-application.md)
- [Changing from v0.0.1](Changing-from-v0_0_1.md)
- examples
    - [MNIST classification](../examples/mnist/readme.md)
    - [Cifar-10 classification](../examples/cifar10/readme.md)
    - [Importing caffe's model](../examples/caffe_converter/readme.md)

## Application Links
I'm willing to update this list if your software use tiny-cnn. Please contact me at Email(see my github profile).

- [zhangqianhui/CnnForAndroid](https://github.com/zhangqianhui/CnnForAndroid) - A Vehicle Recognition Project using Convolutional Neural Network(CNN) in Android platform
- [edgarriba/opencv_contrib](https://github.com/edgarriba/opencv_contrib) (in progress) - A new opencv's dnn module which use tiny-cnn as its backend

## Developer guides
- [How to contribute](CONTRIBUTING.md)
- [Adding a new layer](Adding-a-new-layer.md)

## A quick introduction to tiny-cnn
Include tiny_cnn.h:

```cpp
    #include "tiny_cnn/tiny_cnn.h"
    using namespace tiny_cnn;
    using namespace tiny_cnn::layers;
    using namespace tiny_cnn::activation;
```

Declare the model as ```network```. There are 2 types of network: ```network<sequential>``` and ```network<graph>```. The sequential model is easier to construct.

```cpp
    network<sequential> net;
```

Stack layers:

```cpp
    net << conv<tan_h>(32, 32, 5, 1, 6, padding::same)  // in:32x32x1, 5x5conv, 6fmaps
        << max_pool<tan_h>(32, 32, 6, 2)                // in:32x32x6, 2x2pooling
        << conv<tan_h>(16, 16, 5, 6, 16, padding::same) // in:16x16x6, 5x5conv, 16fmaps 
        << max_pool<tan_h>(16, 16, 16, 2)               // in:16x16x16, 2x2pooling
        << fc<tan_h>(8*8*16, 100)                       // in:8x8x16, out:100
        << fc<softmax>(100, 10);                        // in:100 out:10
```

Some layer takes an activation as a template parameter : ```max_pool<relu>``` means "apply a relu activation after the pooling". if the layer has no successive activation, use ```max_pool<identity>``` instead.

Declare the optimizer:

```cpp
    adagrad opt;
```

In addition to gradient descent, you can use modern optimizers such as adagrad, adadelta, adam.

Now you can start the training:

```cpp
    int epochs = 50;
    int batch = 20;
    net.fit<cross_entropy>(opt, x_data, y_data, batch, epochs);
```

If you don't have the target vector but have the class-id, you can alternatively use ```train```.

```cpp
    net.train<cross_entropy>(opt, x_data, y_label, batch, epochs);
```

Validate the training result:

```cpp
    auto test_result = net.test(x_data, y_label);
    auto loss = net.get_loss<cross_entropy>(x_data, y_data);
```

Generate prediction on the new data:

```cpp
    auto y_vector = net.predict(x_data);
    auto y_label = net.predict_max_label(x_data);
```

Save the trained parameter:

```cpp
    ofstream ofs("data", std::ios::binary);
    ofs << net;
```

For a more in-depth about tiny-cnn, check out [MNIST classification](../examples/mnist/readme.md) where you can see the end-to-end example.
You will find tiny-cnn's API in [How-to](How-Tos.md).

