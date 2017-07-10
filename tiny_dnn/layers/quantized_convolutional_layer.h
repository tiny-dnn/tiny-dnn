/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tiny_dnn/core/backend_tiny.h"
#ifdef CNN_USE_AVX
#include "tiny_dnn/core/backend_avx.h"
#endif

#include "tiny_dnn/util/util.h"

#ifdef DNN_USE_IMAGE_API
#include "tiny_dnn/util/image.h"
#endif

namespace tiny_dnn {

/**
 * 2D convolution layer
 *
 * take input as two-dimensional *image* and applying filtering operation.
 **/
class quantized_convolutional_layer : public layer {
 public:
  /**
   * constructing convolutional layer
   *
   * @param in_width     [in] input image width
   * @param in_height    [in] input image height
   * @param window_size  [in] window(kernel) size of convolution
   * @param in_channels  [in] input image channels (grayscale=1, rgb=3)
   * @param out_channels [in] output image channels
   * @param padding      [in] rounding strategy
   *                          valid: use valid pixels of input only.
   *output-size
   *= (in-width - window_size + 1) *
   *(in-height - window_size + 1) * out_channels
   *                          same: add zero-padding to keep same width/height.
   *output-size = in-width * in-height *
   *out_channels
   * @param has_bias     [in] whether to add a bias vector to the filter
   *outputs
   * @param w_stride     [in] specify the horizontal interval at which to apply
   *the filters to the input
   * @param h_stride     [in] specify the vertical interval at which to apply
   *the
   *filters to the input
   **/
  quantized_convolutional_layer(
    size_t in_width,
    size_t in_height,
    size_t window_size,
    size_t in_channels,
    size_t out_channels,
    padding pad_type             = padding::valid,
    bool has_bias                = true,
    size_t w_stride              = 1,
    size_t h_stride              = 1,
    core::backend_t backend_type = core::backend_t::internal)
    : layer(std_input_order(has_bias), {vector_type::data}) {
    conv_set_params(shape3d(in_width, in_height, in_channels), window_size,
                    window_size, out_channels, pad_type, has_bias, w_stride,
                    h_stride);
    init_backend(backend_type);
  }

  /**
   * constructing convolutional layer
   *
   * @param in_width      [in] input image width
   * @param in_height     [in] input image height
   * @param window_width  [in] window_width(kernel) size of convolution
   * @param window_height [in] window_height(kernel) size of convolution
   * @param in_channels   [in] input image channels (grayscale=1, rgb=3)
   * @param out_channels  [in] output image channels
   * @param padding       [in] rounding strategy
   *                          valid: use valid pixels of input only.
   *output-size
   *= (in-width - window_width + 1) *
   *(in-height - window_height + 1) * out_channels
   *                          same: add zero-padding to keep same width/height.
   *output-size = in-width * in-height *
   *out_channels
   * @param has_bias     [in] whether to add a bias vector to the filter
   *outputs
   * @param w_stride     [in] specify the horizontal interval at which to apply
   *the filters to the input
   * @param h_stride     [in] specify the vertical interval at which to apply
   *the
   *filters to the input
   **/
  quantized_convolutional_layer(
    size_t in_width,
    size_t in_height,
    size_t window_width,
    size_t window_height,
    size_t in_channels,
    size_t out_channels,
    padding pad_type             = padding::valid,
    bool has_bias                = true,
    size_t w_stride              = 1,
    size_t h_stride              = 1,
    core::backend_t backend_type = core::backend_t::internal)
    : layer(std_input_order(has_bias), {vector_type::data}) {
    conv_set_params(shape3d(in_width, in_height, in_channels), window_width,
                    window_height, out_channels, pad_type, has_bias, w_stride,
                    h_stride);
    init_backend(backend_type);
  }

  /**
   * constructing convolutional layer
   *
   * @param in_width         [in] input image width
   * @param in_height        [in] input image height
   * @param window_size      [in] window(kernel) size of convolution
   * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
   * @param out_channels     [in] output image channels
   * @param connection_table [in] definition of connections between in-channels
   *and out-channels
   * @param pad_type         [in] rounding strategy
   *                               valid: use valid pixels of input only.
   *output-size = (in-width - window_size + 1) *
   *(in-height - window_size + 1) * out_channels
   *                               same: add zero-padding to keep same
   *width/height. output-size = in-width * in-height *
   *out_channels
   * @param has_bias         [in] whether to add a bias vector to the filter
   *outputs
   * @param w_stride         [in] specify the horizontal interval at which to
   *apply the filters to the input
   * @param h_stride         [in] specify the vertical interval at which to
   *apply
   *the filters to the input
   **/
  quantized_convolutional_layer(
    size_t in_width,
    size_t in_height,
    size_t window_size,
    size_t in_channels,
    size_t out_channels,
    const core::connection_table &connection_table,
    padding pad_type             = padding::valid,
    bool has_bias                = true,
    size_t w_stride              = 1,
    size_t h_stride              = 1,
    core::backend_t backend_type = core::backend_t::internal)
    : layer(std_input_order(has_bias), {vector_type::data}) {
    conv_set_params(shape3d(in_width, in_height, in_channels), window_size,
                    window_size, out_channels, pad_type, has_bias, w_stride,
                    h_stride, connection_table);
    init_backend(backend_type);
  }

  /**
   * constructing convolutional layer
   *
   * @param in_width         [in] input image width
   * @param in_height        [in] input image height
   * @param window_width     [in] window_width(kernel) size of convolution
   * @param window_height    [in] window_height(kernel) size of convolution
   * @param in_channels      [in] input image channels (grayscale=1, rgb=3)
   * @param out_channels     [in] output image channels
   * @param connection_table [in] definition of connections between in-channels
   *and out-channels
   * @param pad_type         [in] rounding strategy
   *                               valid: use valid pixels of input only.
   *output-size = (in-width - window_size + 1) *
   *(in-height - window_size + 1) * out_channels
   *                               same: add zero-padding to keep same
   *width/height. output-size = in-width * in-height *
   *out_channels
   * @param has_bias         [in] whether to add a bias vector to the filter
   *outputs
   * @param w_stride         [in] specify the horizontal interval at which to
   *apply the filters to the input
   * @param h_stride         [in] specify the vertical interval at which to
   *apply
   *the filters to the input
   **/
  quantized_convolutional_layer(
    size_t in_width,
    size_t in_height,
    size_t window_width,
    size_t window_height,
    size_t in_channels,
    size_t out_channels,
    const core::connection_table &connection_table,
    padding pad_type             = padding::valid,
    bool has_bias                = true,
    size_t w_stride              = 1,
    size_t h_stride              = 1,
    core::backend_t backend_type = core::backend_t::internal)
    : layer(std_input_order(has_bias), {vector_type::data}) {
    conv_set_params(shape3d(in_width, in_height, in_channels), window_width,
                    window_height, out_channels, pad_type, has_bias, w_stride,
                    h_stride, connection_table);
    init_backend(backend_type);
  }

  // move constructor
  quantized_convolutional_layer(
    quantized_convolutional_layer &&other)  // NOLINT
    : layer(std::move(other)),
      params_(std::move(other.params_)),
      cws_(std::move(other.cws_)) {
    init_backend(core::backend_t::internal);
  }

  ///< number of incoming connections for each output unit
  size_t fan_in_size() const override {
    return params_.weight.width_ * params_.weight.height_ * params_.in.depth_;
  }

  ///< number of outgoing connections for each input unit
  size_t fan_out_size() const override {
    return (params_.weight.width_ / params_.w_stride) *
           (params_.weight.height_ / params_.h_stride) * params_.out.depth_;
  }

  /**
   * @param in_data      input vectors of this layer (data, weight, bias)
   * @param out_data     output vectors
   **/
  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    // launch convolutional kernel
    if (in_data.size() == 3) {
      layer::backend_->conv2d_q(in_data, out_data);

    } else if (in_data.size() == 6) {
      layer::backend_->conv2d_eq(in_data, out_data);
    }
  }

  /**
   * return delta of previous layer (delta=\frac{dE}{da}, a=wx in
   *fully-connected layer)
   * @param in_data      input vectors (same vectors as forward_propagation)
   * @param out_data     output vectors (same vectors as forward_propagation)
   * @param out_grad     gradient of output vectors (i-th vector correspond
   *with
   *out_data[i])
   * @param in_grad      gradient of input vectors (i-th vector correspond
   *with
   *in_data[i])
   **/
  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    layer::backend_->conv2d_q(in_data, out_data, out_grad, in_grad);
  }

  std::vector<index3d<size_t>> in_shape() const override {
    if (params_.has_bias) {
      return {params_.in, params_.weight,
              index3d<size_t>(1, 1, params_.out.depth_)};
    } else {
      return {params_.in, params_.weight};
    }
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return {params_.out};
  }

  std::string layer_type() const override { return "q_conv"; }

#ifdef DNN_USE_IMAGE_API
  image<> weight_to_image() const {
    image<> img;
    const size_t border_width = 1;
    const auto pitch          = params_.weight.width_ + border_width;
    const auto width          = params_.out.depth_ * pitch + border_width;
    const auto height         = params_.in.depth_ * pitch + border_width;
    const image<>::intensity_t bg_color = 255;
    const vec_t &W                      = *this->weights()[0];

    img.resize(width, height);
    img.fill(bg_color);

    auto minmax = std::minmax_element(W.begin(), W.end());

    for (size_t r = 0; r < params_.in.depth_; ++r) {
      for (size_t c = 0; c < params_.out.depth_; ++c) {
        if (!params_.tbl.is_connected(c, r)) continue;

        const auto top  = r * pitch + border_width;
        const auto left = c * pitch + border_width;

        size_t idx = 0;

        for (size_t y = 0; y < params_.weight.height_; ++y) {
          for (size_t x = 0; x < params_.weight.width_; ++x) {
            idx             = c * params_.in.depth_ + r;
            idx             = params_.weight.get_index(x, y, idx);
            const float_t w = W[idx];

            img.at(left + x, top + y) = static_cast<image<>::intensity_t>(
              rescale(w, *minmax.first, *minmax.second, 0, 255));
          }
        }
      }
    }
    return img;
  }
#endif  // DNN_USE_IMAGE_API

  friend struct serialization_buddy;

 private:
  void conv_set_params(
    const shape3d &in,
    size_t w_width,
    size_t w_height,
    size_t outc,
    padding ptype,
    bool has_bias,
    size_t w_stride,
    size_t h_stride,
    const core::connection_table &tbl = core::connection_table()) {
    params_.in = in;
    params_.in_padded =
      shape3d(in_length(in.width_, w_width, ptype),
              in_length(in.height_, w_height, ptype), in.depth_);
    params_.out =
      shape3d(conv_out_length(in.width_, w_width, w_stride, ptype),
              conv_out_length(in.height_, w_height, h_stride, ptype), outc);
    params_.weight   = shape3d(w_width, w_height, in.depth_ * outc);
    params_.has_bias = has_bias;
    params_.pad_type = ptype;
    params_.w_stride = w_stride;
    params_.h_stride = h_stride;
    params_.tbl      = tbl;
  }

  void init() {
    if (params_.pad_type == padding::same) {
      cws_.prev_out_buf_.resize(1, vec_t(params_.in_padded.size(), float_t{0}));
      cws_.prev_delta_padded_.resize(
        1, vec_t(params_.in_padded.size(), float_t{0}));
    } else {
      cws_.prev_out_buf_.clear();
    }
  }

  size_t in_length(size_t in_length,
                   size_t window_size,
                   padding pad_type) const {
    return pad_type == padding::same ? (in_length + window_size - 1)
                                     : in_length;
  }

  static size_t conv_out_length(size_t in_length,
                                size_t window_size,
                                size_t stride,
                                padding pad_type) {
    float_t tmp;
    if (pad_type == padding::same) {
      tmp = static_cast<float_t>(in_length) / stride;
    } else if (pad_type == padding::valid) {
      tmp = static_cast<float_t>(in_length - window_size + 1) / stride;
    } else {
      throw nn_error("Not recognized pad_type.");
    }
    return static_cast<size_t>(ceil(tmp));
  }

  static size_t conv_out_dim(size_t in_width,
                             size_t in_height,
                             size_t window_size,
                             size_t w_stride,
                             size_t h_stride,
                             padding pad_type) {
    return conv_out_length(in_width, window_size, w_stride, pad_type) *
           conv_out_length(in_height, window_size, h_stride, pad_type);
  }

  size_t conv_out_dim(size_t in_width,
                      size_t in_height,
                      size_t window_width,
                      size_t window_height,
                      size_t w_stride,
                      size_t h_stride,
                      padding pad_type) const {
    return conv_out_length(in_width, window_width, w_stride, pad_type) *
           conv_out_length(in_height, window_height, h_stride, pad_type);
  }

  void copy_and_pad_input(const tensor_t &in) {
    core::conv_layer_worker_specific_storage &cws = cws_;

    const size_t sample_count = in.size();

    cws.prev_out_padded_.resize(sample_count);

    if (params_.pad_type == padding::same) {
      cws.prev_out_buf_.resize(sample_count, cws.prev_out_buf_[0]);
      cws.prev_delta_padded_.resize(sample_count, cws.prev_delta_padded_[0]);
    }

    for (size_t sample = 0; sample < sample_count; ++sample) {
      if (params_.pad_type == padding::valid) {
        cws.prev_out_padded_[sample] = &(in[sample]);
      } else {
        vec_t *dst = &cws.prev_out_buf_[sample];

        // make padded version in order to avoid corner-case in
        // fprop/bprop
        for (size_t c = 0; c < params_.in.depth_; c++) {
          float_t *pimg = &(*dst)[params_.in_padded.get_index(
            params_.weight.width_ / 2, params_.weight.height_ / 2, c)];
          const float_t *pin = &in[sample][params_.in.get_index(0, 0, c)];

          for (size_t y = 0; y < params_.in.height_; y++,
                      pin += params_.in.width_,
                      pimg += params_.in_padded.width_) {
            std::copy(pin, pin + params_.in.width_, pimg);
          }
        }

        cws.prev_out_padded_[sample] = &(cws.prev_out_buf_[sample]);
      }
    }
  }

  void copy_and_unpad_delta(const tensor_t &delta, tensor_t &delta_unpadded) {
    if (params_.pad_type == padding::valid) {
      delta_unpadded = delta;
    } else {
      for (size_t sample = 0; sample < delta.size(); sample++) {
        size_t idx       = 0;
        const vec_t &src = delta[sample];
        vec_t &dst       = delta_unpadded[sample];

        for (size_t c = 0; c < params_.in.depth_; c++) {
          float_t *pdst = &dst[params_.in.get_index(0, 0, c)];
          idx           = params_.in_padded.get_index(params_.weight.width_ / 2,
                                            params_.weight.height_ / 2, c);
          const float_t *pin = &src[idx];

          for (size_t y = 0; y < params_.in.height_; y++) {
            std::copy(pin, pin + params_.in.width_, pdst);
            pdst += params_.in.width_;
            pin += params_.in_padded.width_;
          }
        }
      }
    }
  }

  void init_backend(const core::backend_t backend_type) {
    std::shared_ptr<core::backend> backend = nullptr;

    // allocate new backend
    if (backend_type == core::backend_t::internal) {
      backend = std::make_shared<core::tiny_backend>(
        &params_, [this](const tensor_t &in) { return copy_and_pad_input(in); },
        [this](const tensor_t &delta, tensor_t &dst) {
          return copy_and_unpad_delta(delta, dst);
        },
        &cws_);
    } else {
      throw nn_error("Not supported backend type.");
    }

    if (backend) {
      layer::set_backend(backend);
      layer::backend_->set_layer(this);
    } else {
      throw nn_error("Could not allocate the backend.");
    }
  }

  /* The convolution parameters */
  core::conv_params params_;

  /* The type of backend */
  // backend_t backend_type_;

  /* Workers buffers */
  core::conv_layer_worker_specific_storage cws_;
};

}  // namespace tiny_dnn
