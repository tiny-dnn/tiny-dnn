# Adding a new layer
This section describes how to create a new layer incorporated with tiny-dnn. Let's create simple fully-connected layer for example.

> Note: This document is old, and doesn't fit to current tiny-dnn. We need to update.

### Declare class
Let's define your layer. All of layer operations in tiny-dnn are derived from ```layer``` class.

```cpp
// calculate y = Wx + b 
class fully_connected : public layer {
public:
    //todo 
};

```

the ```layer``` class prepares input/output data for  your calculation. To do this, you must tell ```layer```'s constructor what you need.

```cpp
layer::layer(const std::vector<vector_type>& in_type,
             const std::vector<vector_type>& out_type)
```

For example, consider calculating fully-connected operation:  ```y = Wx + b```. In this caluculation, Input (right hand of this eq) is data ```x```, weight ```W``` and bias ```b```. Output  is, of course ```y```. So it's constructor should pass {data,weight,bias} as input and {data} as output.

```cpp
// calculate y = Wx + b
class fully_connected : public layer {
public:
    fully_connected(size_t x_size, size_t y_size)
    :layer({vector_type::data,vector_type::weight,vector_type::bias}, // x, W and b
           {vector_type::data}),
     x_size_(x_size),
     y_size_(y_size)
    {}

private:
    size_t x_size_; // number of input elements
    size_t y_size_; // number of output elements
};

```

the ```vector_type::data``` is some input data passed by previous layer, or output data consumed by next layer. ```vector_type::weight``` and ```vector_type::bias``` represents trainable parameters. The only difference between them is default initialization method: ```weight``` is initialized by random value, and ```bias``` is initialized by zero-vector (this behaviour can be changed by network::weight_init method). If you need another vector to calculate, ```vector_type::aux``` can be used.

### Implement virtual method
There are 5 methods to implement. In most case 3 methods are written as one-liner and remaining 2 are essential:

- layer_type
- in_shape
- out_shape
- forward_propagation
- back_propagation

##### layer_type
Returns name of your layer.

```cpp
std::string layer_type() const override {
    return "fully-connected";
}
```

##### in_shape/out_shape
Returns input/output shapes corresponding to inputs/outputs. Shapes is defined by [width, height, depth]. For example fully-connected layer treats input data as 1-dimensional array, so its shape is [N, 1, 1].

```cpp
std::vector<shape3d> in_shape() const override {
    // return input shapes
    // order of shapes must be equal to argument of layer constructor
    return { shape3d(x_size_, 1, 1), // x
             shape3d(x_size_, y_size_, 1), // W
             shape3d(y_size_, 1, 1) }; // b
}

std::vector<shape3d> out_shape() const override {
    return { shape3d(y_size_, 1, 1) }; // y
}
```

#### forward_propagation
Execute forward calculation in this method.

```cpp
void forward_propagation(serial_size_t worker_index,
                         const std::vector<vec_t*>& in_data,
                         std::vector<vec_t*>& out_data) override {
    const vec_t& x = *in_data[0]; // it's size is in_shapes()[0] (=[x_size_,1,1])
    const vec_t& W = *in_data[1];
    const vec_t& b = *in_data[2];
    vec_t& y = *out_data[0];

    std::fill(y.begin(), y.end(), 0.0);

    // y = Wx+b
    for (size_t r = 0; r < y_size_; r++) {
        for (size_t c = 0; c < x_size_; c++)
            y[r] += W[r*x_size_+c]*x[c];
        y[r] += b[r];
    }
}
```

the ```in_data/out_data``` is array of input/output data, which is ordered as you told ```layer```'s constructor. The implementation is simple and straightforward, isn't it?

```worker_index``` is task-id. It is always zero if you run tiny-dnn in single thread. If some class member variables are updated while forward/backward pass, these members must be treated carefully to avoid data race. If their variables are task-independent, your class can hold just N variables and access them by worker_index (you can see this example in [max_pooling_layer.h](../tiny_cnn/layers/max_pooling_layer.h)).
input/output data managed by ```layer``` base class is *task-local*, so ```in_data/out_data``` is treated as if it is running on single thread.

#### back propagation

```cpp
void back_propagation(serial_size_t                index,
                      const std::vector<vec_t*>& in_data,
                      const std::vector<vec_t*>& out_data,
                      std::vector<vec_t*>&       out_grad,
                      std::vector<vec_t*>&       in_grad) override {
    const vec_t& curr_delta = *out_grad[0]; // dE/dy (already calculated in next layer)
    const vec_t& x          = *in_data[0];
    const vec_t& W          = *in_data[1];
    vec_t&       prev_delta = *in_grad[0]; // dE/dx (passed into previous layer)
    vec_t&       dW         = *in_grad[1]; // dE/dW
    vec_t&       db         = *in_grad[2]; // dE/db

    // propagate delta to prev-layer
    for (size_t c = 0; c < x_size_; c++)
        for (size_t r = 0; r < y_size_; r++)
            prev_delta[c] += curr_delta[r] * W[r*x_size_+c];

    // accumulate weight difference
    for (size_t r = 0; r < y_size_; r++)
        for (size_t c = 0; c < x_size_; c++)
            dW[r*x_size_+c] += curr_delta[r] * x[c];

    // accumulate bias difference
    for (size_t r = 0; r < y_size_; r++)
        db[r] += curr_delta[r];
}
```

the ```in_data/out_data``` are just same as forward_propagation, and ```in_grad/out_grad``` are its gradient. Order of gradient values are same as ```in_data/out_data```.

> Note: Gradient of weight/bias are collected over mini-batch and zero-cleared automatically, so you can't use assignment operator to these elements (layer will forget previous training data in mini-batch!). like this example, use ```operator += ``` instead. Gradient of data (```prev_delta``` in the example) may already have meaningful values if two or more layers share this data, so you can't overwrite this value too.

### Verify backward caluculation
It is always a good idea to check if your backward implementation is correct. ```network``` class provides ```gradient_check``` method for this purpose.
Let's add following lines to test/test_network.h and execute test.
```
TEST(network, gradient_check_fully_connected) {
    network<sequential> net;
    net << fully_connected(2, 3)
        << fully_connected(3, 2);

    std::vector<tensor_t> in{ tensor_t{ 1, { 0.5, 1.0 } } };
    std::vector<std::vector<label_t>> t = { std::vector<label_t>(1, {1}) };

    EXPECT_TRUE(net.gradient_check<mse>(in, t, 1e-4, GRAD_CHECK_ALL));
}
```

Congratulations! Now you can use this new class as a tiny-dnn layer.
