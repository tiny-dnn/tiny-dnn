# How-Tos

- [construct network](#construct-network)
    - [sequential model](#sequential-model)
    - [graph model](#graph-model)
- [access layer properties](#access-layer-properties)
- [access weight vector for each layer](#access-weight-vector-for-each-layer)
- [train network](#train-network)
    - [regression](#regression)
    - [classification](#classification)
    - [train graph network](#train-graph-network)
- [visualize each layer activations](#visualize-each-layer-activations)
- [visualize convolution kernels](#visualize-convolution-kernels)
- [save and load network](#save-and-load-network)
- [load dataset](#load-dataset)
- [change weight initialization](#change-weight-initialization)
- [handle error](#handle-error)

    following example codes omits ```using namespace tiny_cnn;```.
    
### <a name="construct-network"></a>construct network
There are two types of network model available: sequential and graph. A graph representation describe network as computational graph - each node of graph is layer, and each directed edge holds tensor and its gradients. Sequential representation describe network as linked list - each layer has at most one predecessor and one successor layer.

Two types of network is represented as network<sequential> and network<graph> class. These two classes have same API, except for its construction.

#### <a name="sequential-model"></a>sequential model
You can construct networks by chaining ```operator <<``` from top(input) to bottom(output).
```cpp
// input: 32x32x1 (1024 dimensions)  output: 10
network<sequential> net;
net << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // 32x32in, conv5x5
    << average_pooling_layer<tan_h>(28, 28, 6, 2) // 28x28in, pool2x2
    << fully_connected_layer<tan_h>(14 * 14 * 6, 120)
    << fully_connected_layer<identity>(120, 10);
```

```cpp
// input: 32x32x3 (3072 dimensions)  output: 40
network<sequential> net;
net << convolutional_layer<relu>(32, 32, 5, 3, 9)
    << average_pooling_layer<relu>(28, 28, 9, 2)
    << fully_connected_layer<tan_h>(14 * 14 * 9, 120)
    << fully_connected_layer<softmax>(120, 40);
```

If you feel these syntax a bit redundant, you can also use "shortcut" names defined in tiny_cnn.h.
```cpp
using namespace tiny_cnn::layers;
net << conv<relu>(32, 32, 5, 3, 9)
    << ave_pool<relu>(28, 28, 9, 2)
    << fc<tan_h>(14 * 14 * 9, 120)
    << fc<softmax>(120, 40);
```

If your network is simple mlp(multi-layer perceptron), you can also use ```make_mlp``` function.
```cpp
auto mynet = make_mlp<tan_h>({ 32 * 32, 300, 10 });
```
It is equivalent to:
```cpp
network<sequential> mynet;
mynet << fully_conneceted_layer<tan_h>(32*32, 300)
      << fully_connectted_layer<tan_h>(300, 10);
```

#### <a name="graph-model"></a>graph model
To construct network which has branch/merge in their model, you can use ```network<graph>``` class. In graph model, you should declare each "node" (layer) at first, and then connect them by ```operator <<```. If two or more nodes are fed into 1 node, ```operator,``` can be used for this purpose.
After connecting all layers, call ```construct_graph``` function to register node connections to graph-network.

```cpp
// declare nodes
layers::input in1(shape3d(3, 1, 1));
layers::input in2(shape3d(3, 1, 1));
layers::add added(2, 3);
layers::fc<relu> out(3, 2);

// connect
(in1, in2) << added;
added << out;

// register to graph
network<graph> net;
construct_graph(net, { &in1, &in2 }, { &out });
```

### <a name="access-layer-properties"></a>access layer properties
You can access each layer by operator[] after construction.

```cpp
...
network<sequential> nn;

nn << convolutional_layer<tan_h>(32, 32, 5, 3, 6)
    << max_pooling_layer<tan_h>(28, 28, 6, 2)
    << fully_connected_layer<tan_h>(14 * 14 * 6, 10);

for (int i = 0; i < nn.depth(); i++) {
    cout << "#layer:" << i << "\n";
    cout << "layer type:" << nn[i]->layer_type() << "\n";
    cout << "input:" << nn[i]->in_data_size() << "(" << nn[i]->in_data_shape() << ")\n";
    cout << "output:" << nn[i]->out_data_size() << "(" << nn[i]->out_data_shape() << ")\n";
}
```

output:
```shell
#layer:0
layer type:conv
input:3072([[32x32x3]])
output:4704([[28x28x6]])
num of parameters:456
#layer:1
layer type:max-pool
input:4704([[28x28x6]])
output:1176([[14x14x6]])
num of parameters:0
#layer:2
layer type:fully-connected
input:1176([[1176x1x1]])
output:10([[10x1x1]])
num of parameters:11770
```

### <a name="access-weight-vector-for-each-layer"></a>access weight vector for each layer
```cpp
std::vector<vec_t*> weights = nn[i]->get_weights();
```
Number of elements differs by layer types and settings. For example, in fully-connected layer with bias term, weights[0] represents weight matrix and weights[1] represents bias vector.

### <a name="train network"></a>train network
#### <a name="regression"></a>regression
Use ```network::fit``` function to train. Specify loss function by template parameter (```mse```, ```cross_entropy```, ```cross_entropy_multiclass``` are available), and fed optimizing algorithm into first argument.
```cpp
network<sequential> net;
adagrad opt;
net << layers::fc<tan_h>(2,3) << layers::fc<softmax>(3,1);

// 2training data, each data type is 2-dimensional array
std::vector<vec_t> input_data  { { 1, 0 }, { 0, 2 } };
std::vector<vec_t> desired_out {    { 2 },    { 1 } };
size_t batch_size = 1;
size_t epochs = 30;

net.fit<mse>(opt, input_data, desired_out, bacth_size, epochs);
```

If you want to do something for each epoch / minibatch (profiling, evaluaing accuracy, saving networks, changing learning rate of optimizer, etc), you can register callbacks for this purpose.

```cpp
// test&save for each epoch
int epoch = 0;
timer t;
nn.fit<mse>(opt, train_images, train_labels, 50, 20,
         // called for each mini-batch
         [&](){
           t.elapsed();
           t.reset();
         },
         // called for each epoch
         [&](){
           result res = nn.test(test_images, test_labels);
           cout << res.num_success << "/" << res.num_total << endl;
           ofstream ofs (("epoch_"+to_string(epoch++)).c_str());
           ofs << nn;
         });
```

#### <a name="classification"></a>classification
As with regression task, you can use ```network::fit``` function to do. Besides, if you have labels(class-id) for each training data, ```network::train``` can be used. Difference between ```network::fit``` and ```network::train``` is how to specify the desired outputs - ```network::train``` takes ```label_t``` type, instead of ```vec_t```.

```cpp
network<sequential> net;
adagrad opt;
net << layers::fc<tan_h>(2,3) << layers::fc<softmax>(3,4);

// input_data[0] should be classified to id:3
// input_data[1] should be classified to id:1
std::vector<vec_t> input_data    { { 1, 0 }, { 0, 2 } };
std::vector<label_t> desired_out {        3,        1 };
size_t batch_size = 1;
size_t epochs = 30;

net.train<mse>(opt, input_data, desired_out, bacth_size, epochs);
```

If you train network as classification task, you must construct model so that output dimension must be greater or equal to (maximum-label-id - 1). So if you train MNIST digit classification (this has 10 classes, labelled by 0-9), the last layer's output dimension must be greater or equal to 10.

#### <a name="train-graph-network"></a>train graph network
If you train graph network, be sure to fed input/output data which has same shape to network's input/output layers.
```cpp
network<graph>    net;
layers::input     in1(2);
layers::input     in2(2);
layers::concat    concat(2, 2);
layers::fc<relu> fc(4, 2);
adagrad opt;

(in1, in2) << concat;
concat << fc;
construct_graph(net, { &in1, &in2 }, { &fc });

// 2training data, each data type is tensor_t and shape is [2x2]
//
//                        1st data for in1       2nd data for in1
//                              |                      |
//                              |   1st data for in2   |   2nd data for in2
//                              |         |            |         |
std::vector<tensor_t> data{ { { 1, 0 }, { 3, 2 } },{ { 0, 2 }, { 1, 1 } } };
std::vector<tensor_t> out { {           { 2, 5 } },{           { 3, 1 } } };

net.fit<mse>(opt, data, out, 1, 1);
```

without callback
```
...
// minibatch=50, epoch=20
nn.train(train_images, train_labels, 50, 20);
```

with callback
```
...
// test&save for each epoch
int epoch = 0;
nn.train(train_images, train_labels, 50, 20, [](){},
         [&](){
           result res = nn.test(test_images, test_labels);
           cout << res.num_success << "/" << res.num_total << endl;
           ofstream ofs (("epoch_"+to_string(epoch++)).c_str());
           ofs << nn;
         });
```

### <a name="visualize-each-layer-activations"></a>visualize each layer activations
```cpp
network<cross_entropy, RMSprop> nn;

nn << convolutional_layer<tan_h>(32, 32, 5, 3, 6)
    << max_pooling_layer<tan_h>(28, 28, 6, 2)
    << fully_connected_layer<tan_h>(14 * 14 * 6, 10);
...
image img = nn[0]->output_to_image(); // visualize activations of recent input
img.write("layer0.bmp");
```

### <a name="visualize-convolution-kernels"></a>visualize convolution kernels
```cpp
network<cross_entropy, RMSprop> nn;

nn << convolutional_layer<tan_h>(32, 32, 5, 3, 6)
    << max_pooling_layer<tan_h>(28, 28, 6, 2)
    << fully_connected_layer<tan_h>(14 * 14 * 6, 10);
...
image img = nn.at<convolutional_layer<tan_h>>(0).weight_to_image();
img.write("kernel0.bmp");
```

### <a name="save and load network"></a>save and load network
Simply use operator << and >> to save/load network weights.


save
```cpp
network<cross_entropy, RMSprop> nn;

nn << convolutional_layer<tan_h>(32, 32, 5, 3, 6)
    << max_pooling_layer<tan_h>(28, 28, 6, 2)
    << fully_connected_layer<tan_h>(14 * 14 * 6, 10);
...
std::ofstream output("nets.txt");
output << nn;
```

load
```cpp
network<cross_entropy, RMSprop> nn;

nn << convolutional_layer<tan_h>(32, 32, 5, 3, 6)
    << max_pooling_layer<tan_h>(28, 28, 6, 2)
    << fully_connected_layer<tan_h>(14 * 14 * 6, 10);
...
std::ifstream input("nets.txt");
input >> nn;
```
*tiny_cnn saves only weights/biases array, not network structure itself*. So you must construct network(same as training time) before loading.

### <a name="load dataset"></a>load dataset
from MNIST idx format
```cpp
vector<vec_t> images;
vector<label_t> labels;
parse_mnist_images("train-images.idx3-ubyte", &images, -1.0, 1.0, 2, 2);
parse_mnist_labels("train-labels.idx1-ubyte", &labels);
```

from cifar-10 binary format
```cpp
vector<vec_t> images;
vector<label_t> labels;
parse_cifar10("data_batch1.bin", &images, &labels, -1.0, 1.0, 0, 0); 
```

### <a name="change-weight-initialization"></a>change weight initialization
In neural network training, initial value of weight/bias can affect training speed and accuracy. In tiny-cnn, the weight is appropriately scaled by xavier algorithm[1](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf) and the bias is filled with 0.

To change initialization method and scaling factor, use ```weight_init()``` and ```bias_init()``` function of network and layer class.

- xavier ... automatic scaling using sqrt(scale / (fan-in + fan-out))
- lecun ... automatic scaling using scale / sqrt(fan-in)
- constant ... fill constant value

```cpp
int num_units [] = { 100, 400, 100 };
auto nn = make_mlp<mse, gradient_descent, tan_h>(num_units, num_units + 3);

// change all layers at once
nn.weight_init(weight_init::lecun());
nn.bias_init(weight_init::xavier(2.0));

// change specific layer
nn[0]->weight_init(weight_init::xavier(4.0));
nn[0]->bias_init(weight_init::constant(1.0));
```
### <a name="handle-error"></a>handle error
tiny_cnn throws ```tiny_cnn::nn_error``` types in run-time. You can see detail error message by ```nn_error::what()``` funtion.

```cpp
try {
   network<cross_entropy, RMSprop> nn;
   ...
} catch (const nn_error& e) {
   cout << e.what();
}
```