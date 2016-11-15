# Layers


<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/arithmetic_layer.h#L36)</span>
## elementwise_add_layer

element-wise add N vectors ```y_i = x0_i + x1_i + ... + xnum_i```

### Constructors

```cpp
    elementwise_add_layer(serial_size_t num_args, serial_size_t dim)
```

- **dim** number of elements for each input

- **num_args** number of inputs

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/average_pooling_layer.h#L136)</span>
## average_pooling_layer

average pooling with trainable weights

### Constructors

```cpp
    average_pooling_layer(serial_size_t in_width,
                          serial_size_t in_height,
                          serial_size_t in_channels,
                          serial_size_t pool_size)
```

- **in_height** height of input image

- **in_channels** the number of input image channels(depth)

- **in_width** width of input image

- **pool_size** factor by which to downscale

```cpp
    average_pooling_layer(serial_size_t in_width,
                          serial_size_t in_height,
                          serial_size_t in_channels,
                          serial_size_t pool_size,
                          serial_size_t stride)
```

- **in_height** height of input image

- **stride** interval at which to apply the filters to the input

- **in_channels** the number of input image channels(depth)

- **in_width** width of input image

- **pool_size** factor by which to downscale

```cpp
    average_pooling_layer(serial_size_t     in_width,
                          serial_size_t     in_height,
                          serial_size_t     in_channels,
                          serial_size_t     pool_size_x,
                          serial_size_t     pool_size_y,
                          serial_size_t     stride_x,
                          serial_size_t     stride_y,
                          padding        pad_type = padding::valid)
```

- **in_height** height of input image

- **pad_type** padding mode(same/valid)

- **in_channels** the number of input image channels(depth)

- **pool_size_x** factor by which to downscale

- **pool_size_y** factor by which to downscale

- **in_width** width of input image

- **stride_x** interval at which to apply the filters to the input

- **stride_y** interval at which to apply the filters to the input

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/average_unpooling_layer.h#L134)</span>
## average_unpooling_layer

average pooling with trainable weights

### Constructors

```cpp
    average_unpooling_layer(serial_size_t in_width,
                            serial_size_t in_height,
                            serial_size_t in_channels,
                            serial_size_t pooling_size)
```

- **in_height** height of input image

- **in_channels** the number of input image channels(depth)

- **in_width** width of input image

- **pooling_size** factor by which to upscale

```cpp
    average_unpooling_layer(serial_size_t in_width,
                            serial_size_t in_height,
                            serial_size_t in_channels,
                            serial_size_t pooling_size,
                            serial_size_t stride)
```

- **in_height** height of input image

- **stride** interval at which to apply the filters to the input

- **in_channels** the number of input image channels(depth)

- **in_width** width of input image

- **pooling_size** factor by which to upscale

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/batch_normalization_layer.h#L42)</span>
## batch_normalization_layer

Batch Normalization

 Normalize the activations of the previous layer at each batch

### Constructors

```cpp
    batch_normalization_layer(const layer& prev_layer,
                              float_t epsilon = 1e-5,
                              float_t momentum = 0.999,
                              net_phase phase = net_phase::train)
```

- **phase** specify the current context (train/test)

- **epsilon** small positive value to avoid zero-division

- **prev_layer** previous layer to be connected with this layer

- **momentum** momentum in the computation of the exponential average of the mean/stddev of the data

```cpp
    batch_normalization_layer(serial_size_t in_spatial_size, 
                              serial_size_t in_channels,                        
                              float_t epsilon = 1e-5,
                              float_t momentum = 0.999,
                              net_phase phase = net_phase::train)
```

- **phase** specify the current context (train/test)

- **in_channels** channels of the input data

- **in_spatial_size** spatial size (WxH) of the input data

- **momentum** momentum in the computation of the exponential average of the mean/stddev of the data

- **epsilon** small positive value to avoid zero-division

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/concat_layer.h#L44)</span>
## concat_layer

concat N layers along depth

 ```cpp
 // in: [3,1,1],[3,1,1] out: [3,1,2] (in W,H,K order)
 concat_layer l1(2,3); 

 // in: [3,2,2],[3,2,5] out: [3,2,7] (in W,H,K order)
 concat_layer l2({shape3d(3,2,2),shape3d(3,2,5)});
 ```

### Constructors

```cpp
    concat_layer(const std::vector<shape3d>& in_shapes)
```

- **in_shapes** shapes of input tensors

```cpp
    concat_layer(serial_size_t num_args, serial_size_t ndim)
```

- **ndim** number of elements for each input

- **num_args** number of input tensors

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/convolutional_layer.h#L52)</span>
## convolutional_layer

2D convolution layer

 take input as two-dimensional *image* and applying filtering operation.

### Constructors

```cpp
    convolutional_layer(serial_size_t in_width,
                        serial_size_t in_height,
                        serial_size_t window_size,
                        serial_size_t in_channels,
                        serial_size_t out_channels,
                        padding    pad_type = padding::valid,
                        bool       has_bias = true,
                        serial_size_t w_stride = 1,
                        serial_size_t h_stride = 1,
                        backend_t  backend_type = core::default_engine())
```

- **in_height** input image height

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **window_size** window(kernel) size of convolution

- **has_bias** whether to add a bias vector to the filter outputs

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **padding** rounding strategy
  - valid: use valid pixels of input only. ```output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels```
  - same: add zero-padding to keep same width/height. ```output-size = in-width * in-height * out_channels```

- **in_channels** input image channels (grayscale=1, rgb=3)

- **backend_type** specify backend engine you use

- **in_width** input image width

```cpp
    convolutional_layer(serial_size_t in_width,
                        serial_size_t in_height,
                        serial_size_t window_width,
                        serial_size_t window_height,
                        serial_size_t in_channels,
                        serial_size_t out_channels,
                        padding    pad_type = padding::valid,
                        bool       has_bias = true,
                        serial_size_t w_stride = 1,
                        serial_size_t h_stride = 1,
                        backend_t  backend_type = core::default_engine())
```

- **in_height** input image height

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **backend_type** specify backend engine you use

- **has_bias** whether to add a bias vector to the filter outputs

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **window_height** window_height(kernel) size of convolution

- **window_width** window_width(kernel) size of convolution

- **padding** rounding strategy
  - valid: use valid pixels of input only. ```output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels```
  - same: add zero-padding to keep same width/height. ```output-size = in-width * in-height * out_channels```

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

```cpp
    convolutional_layer(serial_size_t              in_width,
                        serial_size_t              in_height,
                        serial_size_t              window_size,
                        serial_size_t              in_channels,
                        serial_size_t              out_channels,
                        const connection_table& connection_table,
                        padding                 pad_type = padding::valid,
                        bool                    has_bias = true,
                        serial_size_t              w_stride = 1,
                        serial_size_t              h_stride = 1,
                        backend_t      backend_type = core::default_engine())
```

- **in_height** input image height

- **window_size** window(kernel) size of convolution

- **has_bias** whether to add a bias vector to the filter outputs

- **connection_table** definition of connections between in-channels and out-channels

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **pad_type** rounding strategy
  - valid: use valid pixels of input only. ```output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels```
  - same: add zero-padding to keep same width/height. ```output-size = in-width * in-height * out_channels```

- **in_channels** input image channels (grayscale=1, rgb=3)

- **backend_type** specify backend engine you use

- **in_width** input image width

```cpp
    convolutional_layer(serial_size_t              in_width,
                        serial_size_t              in_height,
                        serial_size_t              window_width,
                        serial_size_t              window_height,
                        serial_size_t              in_channels,
                        serial_size_t              out_channels,
                        const connection_table& connection_table,
                        padding                 pad_type = padding::valid,
                        bool                    has_bias = true,
                        serial_size_t              w_stride = 1,
                        serial_size_t              h_stride = 1,
                        backend_t      backend_type = core::default_engine())
```

- **in_height** input image height

- **backend_type** specify backend engine you use

- **has_bias** whether to add a bias vector to the filter outputs

- **connection_table** definition of connections between in-channels and out-channels

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **window_height** window_height(kernel) size of convolution

- **window_width** window_width(kernel) size of convolution

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **pad_type** rounding strategy
  - valid: use valid pixels of input only. ```output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels```
  - same: add zero-padding to keep same width/height. ```output-size = in-width * in-height * out_channels```

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/deconvolutional_layer.h#L54)</span>
## deconvolutional_layer

2D deconvolution layer

 take input as two-dimensional *image* and applying filtering operation.

### Constructors

```cpp
    deconvolutional_layer(serial_size_t     in_width,
                          serial_size_t     in_height,
                          serial_size_t     window_size,
                          serial_size_t     in_channels,
                          serial_size_t     out_channels,
                          padding        pad_type = padding::valid,
                          bool           has_bias = true,
                          serial_size_t     w_stride = 1,
                          serial_size_t     h_stride = 1,
                          backend_t      backend_type = core::default_engine())
```

- **in_height** input image height

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **window_size** window(kernel) size of convolution

- **has_bias** whether to add a bias vector to the filter outputs

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **padding** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

```cpp
    deconvolutional_layer(serial_size_t     in_width,
                          serial_size_t     in_height,
                          serial_size_t     window_width,
                          serial_size_t     window_height,
                          serial_size_t     in_channels,
                          serial_size_t     out_channels,
                          padding        pad_type = padding::valid,
                          bool           has_bias = true,
                          serial_size_t     w_stride = 1,
                          serial_size_t     h_stride = 1,
                          backend_t      backend_type = core::default_engine())
```

- **in_height** input image height

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **has_bias** whether to add a bias vector to the filter outputs

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **window_height** window_height(kernel) size of convolution

- **window_width** window_width(kernel) size of convolution

- **padding** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

```cpp
    deconvolutional_layer(serial_size_t              in_width,
                          serial_size_t              in_height,
                          serial_size_t              window_size,
                          serial_size_t              in_channels,
                          serial_size_t              out_channels,
                          const connection_table& connection_table,
                          padding                 pad_type = padding::valid,
                          bool                    has_bias = true,
                          serial_size_t              w_stride = 1,
                          serial_size_t              h_stride = 1,
                          backend_t               backend_type = core::default_engine())
```

- **in_height** input image height

- **window_size** window(kernel) size of convolution

- **has_bias** whether to add a bias vector to the filter outputs

- **connection_table** definition of connections between in-channels and out-channels

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **pad_type** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

```cpp
    deconvolutional_layer(serial_size_t              in_width,
                          serial_size_t              in_height,
                          serial_size_t              window_width,
                          serial_size_t              window_height,
                          serial_size_t              in_channels,
                          serial_size_t              out_channels,
                          const connection_table& connection_table,
                          padding                 pad_type = padding::valid,
                          bool                    has_bias = true,
                          serial_size_t              w_stride = 1,
                          serial_size_t              h_stride = 1,
                          backend_t               backend_type = core::default_engine())
```

- **in_height** input image height

- **has_bias** whether to add a bias vector to the filter outputs

- **connection_table** definition of connections between in-channels and out-channels

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **window_height** window_height(kernel) size of convolution

- **window_width** window_width(kernel) size of convolution

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **pad_type** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/dropout_layer.h#L37)</span>
## dropout_layer

applies dropout to the input

### Constructors

```cpp
    dropout_layer(serial_size_t in_dim, float_t dropout_rate, net_phase phase = net_phase::train)
```

- **phase** initial state of the dropout

- **dropout_rate** (0-1) fraction of the input units to be dropped

- **in_dim** number of elements of the input

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/feedforward_layer.h#L37)</span>
## feedforward_layer

single-input, single-output network with activation function

### Constructors

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/fully_connected_layer.h#L39)</span>
## fully_connected_layer

compute fully-connected(matmul) operation

### Constructors

```cpp
    fully_connected_layer(serial_size_t in_dim,
                          serial_size_t out_dim,
                          bool       has_bias = true,
                          backend_t  backend_type = core::default_engine())
```

- **out_dim** number of elements of the output

- **has_bias** whether to include additional bias to the layer

- **in_dim** number of elements of the input

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/input_layer.h#L32)</span>
## input_layer

### Constructors

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/linear_layer.h#L38)</span>
## linear_layer

element-wise operation: ```f(x) = h(scale*x+bias)```

### Constructors

```cpp
 linear_layer(serial_size_t dim, float_t scale = float_t(1)
```

- **dim** number of elements

- **scale** factor by which to multiply

- **bias** bias term

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/lrn_layer.h#L42)</span>
## lrn_layer

local response normalization

### Constructors

```cpp
    lrn_layer(layer*      prev,
              serial_size_t  local_size,
              float_t     alpha = 1.0,
              float_t     beta  = 5.0,
              norm_region region = norm_region::across_channels)
```

- **beta** the scaling parameter (same to caffe's LRN)

- **alpha** the scaling parameter (same to caffe's LRN)

- **layer** the previous layer connected to this

- **in_channels** the number of channels of input data

- **local_size** the number of channels(depths) to sum over

```cpp
    lrn_layer(serial_size_t  in_width,
              serial_size_t  in_height,
              serial_size_t  local_size,
              serial_size_t  in_channels,
              float_t     alpha = 1.0,
              float_t     beta  = 5.0,
              norm_region region = norm_region::across_channels)
```

- **in_height** the height of input data

- **local_size** the number of channels(depths) to sum over

- **beta** the scaling parameter (same to caffe's LRN)

- **in_channels** the number of channels of input data

- **alpha** the scaling parameter (same to caffe's LRN)

- **in_width** the width of input data

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/max_pooling_layer.h#L53)</span>
## max_pooling_layer

### Constructors

```cpp
    max_pooling_layer(serial_size_t in_width,
                      serial_size_t in_height,
                      serial_size_t in_channels,
                      serial_size_t pooling_size,
                      backend_t  backend_type = core::default_engine())
```

- **in_height** height of input image

- **in_channels** the number of input image channels(depth)

- **in_width** width of input image

- **pooling_size** factor by which to downscale

```cpp
    max_pooling_layer(serial_size_t in_width,
                      serial_size_t in_height,
                      serial_size_t in_channels,
                      serial_size_t pooling_size_x,
                      serial_size_t pooling_size_y,
                      serial_size_t stride_x,
                      serial_size_t stride_y,
                      padding    pad_type = padding::valid,
                      backend_t  backend_type = core::default_engine())
```

- **in_height** height of input image

- **stride** interval at which to apply the filters to the input

- **in_channels** the number of input image channels(depth)

- **in_width** width of input image

- **pooling_size** factor by which to downscale

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/max_unpooling_layer.h#L38)</span>
## max_unpooling_layer

### Constructors

```cpp
    max_unpooling_layer(serial_size_t in_width,
                        serial_size_t in_height,
                        serial_size_t in_channels,
                        serial_size_t unpooling_size)
```

- **in_height** height of input image

- **in_channels** the number of input image channels(depth)

- **in_width** width of input image

- **unpooling_size** factor by which to upscale

```cpp
    max_unpooling_layer(serial_size_t in_width,
                        serial_size_t in_height,
                        serial_size_t in_channels,
                        serial_size_t unpooling_size,
                        serial_size_t stride)
```

- **in_height** height of input image

- **stride** interval at which to apply the filters to the input

- **in_channels** the number of input image channels(depth)

- **in_width** width of input image

- **unpooling_size** factor by which to upscale

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/partial_connected_layer.h#L34)</span>
## partial_connected_layer

### Constructors

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/power_layer.h#L38)</span>
## power_layer

element-wise pow: ```y = scale*x^factor```

### Constructors

```cpp
    power_layer(const shape3d& in_shape, float_t factor, float_t scale=1.0f)
```

- **factor** floating-point number that specifies a power

- **scale** scale factor for additional multiply

- **in_shape** shape of input tensor

```cpp
    power_layer(const layer& prev_layer, float_t factor, float_t scale=1.0f)
```

- **prev_layer** previous layer to be connected

- **scale** scale factor for additional multiply

- **factor** floating-point number that specifies a power

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/quantized_convolutional_layer.h#L54)</span>
## quantized_convolutional_layer

2D convolution layer

 take input as two-dimensional *image* and applying filtering operation.

### Constructors

```cpp
    quantized_convolutional_layer(serial_size_t     in_width,
                                  serial_size_t     in_height,
                                  serial_size_t     window_size,
                                  serial_size_t     in_channels,
                                  serial_size_t     out_channels,
                                  padding        pad_type = padding::valid,
                                  bool           has_bias = true,
                                  serial_size_t     w_stride = 1,
                                  serial_size_t     h_stride = 1,
                                  backend_t      backend_type = core::backend_t::internal)
```

- **in_height** input image height

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **window_size** window(kernel) size of convolution

- **has_bias** whether to add a bias vector to the filter outputs

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **padding** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

```cpp
    quantized_convolutional_layer(serial_size_t     in_width,
                                  serial_size_t     in_height,
                                  serial_size_t     window_width,
                                  serial_size_t     window_height,
                                  serial_size_t     in_channels,
                                  serial_size_t     out_channels,
                                  padding        pad_type = padding::valid,
                                  bool           has_bias = true,
                                  serial_size_t     w_stride = 1,
                                  serial_size_t     h_stride = 1,
                                  backend_t      backend_type = core::backend_t::internal)
```

- **in_height** input image height

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **has_bias** whether to add a bias vector to the filter outputs

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **window_height** window_height(kernel) size of convolution

- **window_width** window_width(kernel) size of convolution

- **padding** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

```cpp
    quantized_convolutional_layer(serial_size_t              in_width,
                                  serial_size_t              in_height,
                                  serial_size_t              window_size,
                                  serial_size_t              in_channels,
                                  serial_size_t              out_channels,
                                  const connection_table& connection_table,
                                  padding                 pad_type = padding::valid,
                                  bool                    has_bias = true,
                                  serial_size_t              w_stride = 1,
                                  serial_size_t              h_stride = 1,
                                  backend_t backend_type = core::backend_t::internal)
```

- **in_height** input image height

- **window_size** window(kernel) size of convolution

- **has_bias** whether to add a bias vector to the filter outputs

- **connection_table** definition of connections between in-channels and out-channels

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **pad_type** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

```cpp
    quantized_convolutional_layer(serial_size_t              in_width,
                                  serial_size_t              in_height,
                                  serial_size_t              window_width,
                                  serial_size_t              window_height,
                                  serial_size_t              in_channels,
                                  serial_size_t              out_channels,
                                  const connection_table& connection_table,
                                  padding                 pad_type = padding::valid,
                                  bool                    has_bias = true,
                                  serial_size_t              w_stride = 1,
                                  serial_size_t              h_stride = 1,
                                  backend_t      backend_type = core::backend_t::internal)
```

- **in_height** input image height

- **has_bias** whether to add a bias vector to the filter outputs

- **connection_table** definition of connections between in-channels and out-channels

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **window_height** window_height(kernel) size of convolution

- **window_width** window_width(kernel) size of convolution

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **pad_type** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/quantized_deconvolutional_layer.h#L54)</span>
## quantized_deconvolutional_layer

2D deconvolution layer

 take input as two-dimensional *image* and applying filtering operation.

### Constructors

```cpp
    quantized_deconvolutional_layer(serial_size_t     in_width,
                                    serial_size_t     in_height,
                                    serial_size_t     window_size,
                                    serial_size_t     in_channels,
                                    serial_size_t     out_channels,
                                    padding        pad_type = padding::valid,
                                    bool           has_bias = true,
                                    serial_size_t     w_stride = 1,
                                    serial_size_t     h_stride = 1,
                                    backend_t      backend_type = core::backend_t::internal)
```

- **in_height** input image height

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **window_size** window(kernel) size of convolution

- **has_bias** whether to add a bias vector to the filter outputs

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **padding** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

```cpp
    quantized_deconvolutional_layer(serial_size_t     in_width,
                                    serial_size_t     in_height,
                                    serial_size_t     window_width,
                                    serial_size_t     window_height,
                                    serial_size_t     in_channels,
                                    serial_size_t     out_channels,
                                    padding        pad_type = padding::valid,
                                    bool           has_bias = true,
                                    serial_size_t     w_stride = 1,
                                    serial_size_t     h_stride = 1,
                                    backend_t      backend_type = core::backend_t::internal)
```

- **in_height** input image height

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **has_bias** whether to add a bias vector to the filter outputs

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **window_height** window_height(kernel) size of convolution

- **window_width** window_width(kernel) size of convolution

- **padding** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_width + 1) * (in-height - window_height + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

```cpp
    quantized_deconvolutional_layer(serial_size_t              in_width,
                                    serial_size_t              in_height,
                                    serial_size_t              window_size,
                                    serial_size_t              in_channels,
                                    serial_size_t              out_channels,
                                    const connection_table& connection_table,
                                    padding                 pad_type = padding::valid,
                                    bool                    has_bias = true,
                                    serial_size_t              w_stride = 1,
                                    serial_size_t              h_stride = 1,
                                    backend_t               backend_type = core::backend_t::internal)
```

- **in_height** input image height

- **window_size** window(kernel) size of convolution

- **has_bias** whether to add a bias vector to the filter outputs

- **connection_table** definition of connections between in-channels and out-channels

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **pad_type** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

```cpp
    quantized_deconvolutional_layer(serial_size_t              in_width,
                                    serial_size_t              in_height,
                                    serial_size_t              window_width,
                                    serial_size_t              window_height,
                                    serial_size_t              in_channels,
                                    serial_size_t              out_channels,
                                    const connection_table& connection_table,
                                    padding                 pad_type = padding::valid,
                                    bool                    has_bias = true,
                                    serial_size_t              w_stride = 1,
                                    serial_size_t              h_stride = 1,
                                    backend_t               backend_type = core::backend_t::internal)
```

- **in_height** input image height

- **has_bias** whether to add a bias vector to the filter outputs

- **connection_table** definition of connections between in-channels and out-channels

- **out_channels** output image channels

- **w_stride** specify the horizontal interval at which to apply the filters to the input

- **window_height** window_height(kernel) size of convolution

- **window_width** window_width(kernel) size of convolution

- **h_stride** specify the vertical interval at which to apply the filters to the input

- **pad_type** rounding strategy
  valid: use valid pixels of input only. output-size = (in-width - window_size + 1) * (in-height - window_size + 1) * out_channels
  same: add zero-padding to keep same width/height. output-size = in-width * in-height * out_channels

- **in_channels** input image channels (grayscale=1, rgb=3)

- **in_width** input image width

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/quantized_fully_connected_layer.h#L37)</span>
## quantized_fully_connected_layer

compute fully-connected(matmul) operation

### Constructors

```cpp
    quantized_fully_connected_layer(serial_size_t in_dim,
                                    serial_size_t out_dim,
                                    bool       has_bias = true,
                                    backend_t  backend_type = core::backend_t::internal)
```

- **out_dim** number of elements of the output

- **has_bias** whether to include additional bias to the layer

- **in_dim** number of elements of the input

<span style="float:right;">[[source]](https://github.com/tiny-dnn/tiny-dnn/blob/master/tiny_dnn/layers/slice_layer.h#L42)</span>
## slice_layer

slice an input data into multiple outputs along a given slice dimension.

### Constructors

```cpp
    slice_layer(const shape3d& in_shape, slice_type slice_type, serial_size_t num_outputs)
```

- **num_outputs** number of output layers

  example1:
  input:       NxKxWxH = 4x3x2x2  (N:batch-size, K:channels, W:width, H:height)
  slice_type:  slice_samples
  num_outputs: 3

  output[0]: 1x3x2x2
  output[1]: 1x3x2x2
  output[2]: 2x3x2x2  (mod data is assigned to the last output)

  example2:
  input:       NxKxWxH = 4x6x2x2
  slice_type:  slice_channels
  num_outputs: 3

  output[0]: 4x2x2x2
  output[1]: 4x2x2x2
  output[2]: 4x2x2x2

- **slice_type** target axis of slicing


