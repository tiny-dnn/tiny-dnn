// Copyright (c) 2017, Taiga Nomi
#pragma once

#include "tiny_dnn/layers/layers.h"

namespace tiny_dnn {

template <typename T>
void register_layers(T* h) {
  h->template register_layer<average_pooling_layer>("avepool");
  h->template register_layer<batch_normalization_layer>("batchnorm");
  h->template register_layer<concat_layer>("concat");
  h->template register_layer<convolutional_layer>("conv");
  h->template register_layer<dropout_layer>("dropout");
  h->template register_layer<fully_connected_layer>("fully_connected");
  h->template register_layer<global_average_pooling_layer>("global_average_pooling");
  h->template register_layer<linear_layer>("linear");
  h->template register_layer<lrn_layer>("lrn");
  h->template register_layer<max_pooling_layer>("maxpool");
  h->template register_layer<power_layer>("power");
  h->template register_layer<slice_layer>("slice");
  h->template register_layer<elementwise_add_layer>("elementwise_add");

  h->template register_layer<sigmoid_layer>("sigmoid");
  h->template register_layer<tanh_layer>("tanh");
  h->template register_layer<relu_layer>("relu");
  h->template register_layer<softmax_layer>("softmax");
  h->template register_layer<leaky_relu_layer>("leaky_relu");
  h->template register_layer<elu_layer>("elu");
  h->template register_layer<tanh_p1m2_layer>("tanh_scaled");
}

}  // namespace tiny_dnn
