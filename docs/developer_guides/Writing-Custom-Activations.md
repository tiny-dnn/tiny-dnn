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
