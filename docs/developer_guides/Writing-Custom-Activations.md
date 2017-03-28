# Writing Custom Activations

This section describes the procedure to extend tiny-dnn and write custom 
activation functions. Activations are implemented as separate layers in 
tiny-dnn. All the current activation classes inherit ``activation_layer``
class. To form a new activation function, one must create a class and make
it inherit ``activation_layer``.

### Declare class

Let's define a custom activation layer. ``activation_layer`` already provides 
five types of constructors - which allow setting the dimensions of layer 
through constructor arguments. 

```cpp
class my_activation_layer : public activation_layer {
public:
    using activation_layer::activation_layer; 
};

```

If your activation layer has its own variables, say a scalar `alpha`, then you 
need to form your own constructors. ``activation_layer`` already has a member 
named ``in_shape_`` of ``shape3d`` type, to store the dimensions of layer.

```cpp
class my_activation_layer : public activation_layer {
public:
    my_activation_layer(const float_t alpha = 1.0,
                        const shape3d& in_shape)
        : alpha_(alpha), activation_layer(in_shape) {};

    // todo ...
    
private:
    float_t alpha;
```


### Overriding Virtual Methods

The ``activation_layer`` class has four virtual methods which every child 
class must override. They are the ones outlining the behaviour of our custom 
activation.

#### 1. Overriding ``layer_type``

This method returns a string, representing name of the current activation 
function.

```cpp
std::string layer_type() const override {
  return "my-custom-activation";
}
```

#### 2. Overriding ``forward_activation``

This method consists the main logical of our activation function. It takes 
in two vectors passed by reference and fills the second one by applying 
activation function to the first one.

For example, let our activation function be simply a scalar multiplication. 
The implementation would look like:

```cpp
void forward_activation(const vec_t &x, vec_t &y) override {
  for (serial_size_t j = 0; j < x.size(); j++) {
    y[j] = alpha * x[j];
  }
}
```

We could have easily accepted ``float_t`` arguments and applied activation 
function on one element. But this function would have been called for each 
neuron of the layer. **Calling a virtual function inside a tight for loop 
hurts performance.** Hence this is how the method is implemented.  
Practically, each ``vec_t`` here will represent a single flattened image out 
of the minibatch of a particular epoch. 
