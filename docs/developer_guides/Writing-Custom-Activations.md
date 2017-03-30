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

    float_t get_alpha() { return alpha; }

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

This method contains the main logic of our activation function. It takes
in two vectors passed by reference and fills the second one by applying 
activation function to the first one.

For example, let our activation function be simply a scalar multiplication.
The implementation would look like:

```cpp
void forward_activation(const vec_t &x, vec_t &y) override {
  for (size_t j = 0; j < x.size(); j++) {
    y[j] = alpha * x[j];
  }
}
```

We could have easily accepted ``float_t`` arguments and applied activation 
function on one element. But this function would have been called for each 
neuron of the layer. **Calling a virtual function inside a tight for loop 
hurts performance.** Hence this is how the method is implemented.  
Practically, each ``vec_t`` here will represent a single flattened Tensor out
of the minibatch of a particular epoch. 

#### 3. Overriding ``backward_activation``

This method contains the backward gradient flow of our activation. Gradients
of outputs are accepted as input, along with corresponding output and input
vectors. Gradients of input are filled in-place.

For example, the ``backward_activation`` method for our activation function
would look like:

```cpp
void backward_activation(const vec_t &x,
                         const vec_t &y,
                         vec_t &dx,
                         const vec_t &dy) override {
  for (size_t j = 0; j < x.size(); j++) {
    // dx = dy * (gradient of my activation)
    dx[j] = dy[j] * alpha;
  }
}
```

#### 4. Overriding ``scale``

This method returns a pair of ``float_t``, denoting the range of target value
for learning.

```cpp
std::pair<float_t, float_t> scale() const override {
  return std::make_pair(float_t(0.1), float_t(0.9));
};
```

That's it ! Your new activation is now ready as a layer of the network. You can
use it easily as:

```cpp
network<sequential> net;

net << fully_connected_layer(256, 64) << my_activation_layer(64);

// specifying input dimensions is optional if activation layer is not the first
// layer of our network
```


**Note:** The information further is optional, if you wish to do some rough temporary
prototyping, you can skip the following content.


### Register for Serialization

If you wish to serialize your activation layer, you must add these lines to your
class implementation:

```cpp
#ifndef CNN_NO_SERIALIZATION
  friend struct serialization_buddy;
#endif
```

Register a macro at [tiny-dnn/util/serialization_layer_list.h](
https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/util/serialization_layer_list.h)
just like other layers are listed.

#### 1. Create two structs in serialization_functions.h

Both of these should go in ``cereal`` namespace.
```cpp
template <>
struct LoadAndConstruct<tiny_dnn::my_activation_layer> {
  template <class Archive>
  static void load_and_construct(
    Archive &ar, cereal::construct<tiny_dnn::my_activation_layer> &construct) {
    tiny_dnn::shape3d in_shape;
    float_t alpha;

    ar(cereal::make_nvp("in_size", in_shape));
    ar(cereal::make_nvp("alpha", alpha);
    construct(in_shape, alpha);
  }
};

template <class Archive>
struct specialize<Archive,
                  tiny_dnn::my_activation_layer,
                  cereal::specialization::non_member_serialize> {};
```

#### 2. Add a method in ``serialization_buddy`` struct

```cpp
template <class Archive>
static inline void serialize(Archive &ar, tiny_dnn::my_activation_layer &layer) {
  layer.serialize_prolog(ar);
  ar(cereal::make_nvp("in_size", layer.in_shape()[0]));
  ar(cereal::make_nvp("alpha", layer.get_alpha()));
}
```

#### 3. Add a wrapper method in ``serialization_functions.h`` as well

```cpp
template <class Archive>
void serialize(Archive &ar, tiny_dnn::my_activation_layer &layer) {
  serialization_buddy::serialize(ar, layer);
}
```

Now you can get your layer represented in JSON structure of the network, if serialized
by serialization helpers of tiny-dnn.
