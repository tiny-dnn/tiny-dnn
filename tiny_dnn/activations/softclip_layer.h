/*
    Copyright (c) 2017, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <string>
#include <utility>

#include "tiny_dnn/activations/activation_layer.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

class softclip_layer : public activation_layer {
 public:
  // using activation_layer::activation_layer;
  /**
   * Construct a softclip which will take shape when connected to some
   * layer. Connection happens like ( layer1 << act_layer1 ) and shape of this
   * layer is inferred at that time.
   */
  explicit softclip_layer(const float_t alpha )
    : softclip_layer(shape3d(0, 0, 0), alpha ) {}

  /**
   * Construct a flat softclip with specified number of neurons.
   * This constructor is suitable for adding an activation layer after
   * flat layers such as fully connected layers.
   *
   * @param in_dim      [in] number of elements of the input
   */
  softclip_layer(size_t in_dim,
                 const float_t alpha      = 1.0)
    : softclip_layer(shape3d(in_dim, 1, 1), alpha ) {}

  /**
   * Construct a softclip with specified width, height and channels.
   * This constructor is suitable for adding an activation layer after spatial
   * layers such as convolution / pooling layers.
   *
   * @param in_width    [in] number of input elements along width
   * @param in_height   [in] number of input elements along height
   * @param in_channels [in] number of channels (input elements along depth)
   */
  softclip_layer(size_t in_width,
                 size_t in_height,
                 size_t in_channels,
                 const float_t alpha      = 1.0)
    : softclip_layer(
        shape3d(in_width, in_height, in_channels), alpha ) {}

  /**
   * Construct a softclip layer with specified input shape.
   *
   * @param in_shape [in] shape of input tensor
   */
  softclip_layer(const shape3d &in_shape,
                 const float_t alpha      = 1.0)
    : activation_layer(in_shape), alpha_(alpha) {

    alpha_exp_      = exp( alpha_);
    alpha_invert_ = 1/ alpha_;
    threshold_precision_= -9;
    auto epsilon = pow(10.0,threshold_precision_); 
    threshold_left_ = alpha_invert_* log( ( exp( alpha_ *    epsilon )  - 1 ) /( 1 - exp( alpha_ * (   epsilon  -1 ) ) ) );
    threshold_right_= alpha_invert_* log( ( exp( alpha_ * (1-epsilon) ) - 1 )/( 1 - exp( alpha_ * ((1-epsilon) -1 ) ) ) );
  }

  /**
   * Construct a softclip layer given the previous layer.
   * @param prev_layer previous layer
   */
  softclip_layer(const layer &prev_layer,
                 const float_t alpha      = 1.0)
    : activation_layer(prev_layer), alpha_(alpha) {

    alpha_exp_      = exp( alpha_);
    alpha_invert_ = 1/ alpha_;
    threshold_precision_= -9;
    auto epsilon = pow(10.0,threshold_precision_); 
    threshold_left_ = alpha_invert_* log( ( exp( alpha_ *    epsilon )  - 1 ) /( 1 - exp( alpha_ * (   epsilon  -1 ) ) ) );
    threshold_right_= alpha_invert_* log( ( exp( alpha_ * (1-epsilon) ) - 1 )/( 1 - exp( alpha_ * ((1-epsilon) -1 ) ) ) );
  }

  std::string layer_type() const override { return "softclip-activation"; }

  float_t alpha_value() const { return alpha_; }

  void forward_activation(const vec_t &x, vec_t &y) override {
    for (size_t j = 0; j < x.size(); j++) {

      auto const & coord_x =  x[j];
      auto       & coord_y =  y[j];

      if( threshold_right_ < coord_x  ){ coord_y  = float_t(1); continue;} 
      if(  coord_x < threshold_left_  ){ coord_y  = float_t(0); continue;} 

      float_t left   = std::log1p(std::exp(alpha_ *   coord_x));
      float_t right  = std::log1p(std::exp(alpha_ * ( coord_x-float_t(1) )));

      coord_y           =  alpha_invert_ * ( left - right );
    }
  }

  void backward_activation(const vec_t &x,
                           const vec_t &y,
                           vec_t &dx,
                           const vec_t &dy) override {
    for (size_t j = 0; j < x.size(); j++) {
      auto const & coord_x =  x[j];
      auto const & coord_y =  y[j];

      if( threshold_right_ < coord_x  ){ dx[j]  = float_t(0); continue;} 
      if(  coord_x < threshold_left_  ){ dx[j]  = float_t(0); continue;} 

      float_t exp_ax   = std::exp(   alpha_ * coord_x );
      float_t exp_max  = std::exp( - alpha_ * coord_x );

      float_t left  = exp_ax/(1+exp_ax);
      float_t right = exp_ax/(alpha_exp_+exp_ax);

      dx[j] = dy[j] * ( left - right );
    }
  }

  std::pair<float_t, float_t> scale() const override {
    return std::make_pair(float_t(0.1), float_t(0.9));
  }

  float_t alpha_;
  float_t alpha_exp_, alpha_invert_;
  float_t threshold_precision_=9.0; //!< Internal variable. Number of nines or zeros to calculate before function becom constant.
  float_t threshold_left_;
  float_t threshold_right_;
  friend struct serialization_buddy;
};

}  // namespace tiny_dnn
