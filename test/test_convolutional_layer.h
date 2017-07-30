/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <gtest/gtest.h>

#include <vector>

#include "test/testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(convolutional, setup_internal) {
  convolutional_layer l(5, 5, 3, 1, 2, padding::valid, true);

  EXPECT_EQ(l.inputs().size(), 1u);              // num of input edges
  EXPECT_EQ(l.outputs().size(), 1u);             // num of output edges
  EXPECT_EQ(l.in_data_size(), 25u);              // size of input tensors
  EXPECT_EQ(l.out_data_size(), 18u);             // size of output tensors
  EXPECT_EQ(l.fan_in_size(), 9u);                // num of incoming connections
  EXPECT_EQ(l.fan_out_size(), 18u);              // num of outgoing connections
  EXPECT_EQ(l.ith_parameter(0).size(), 18u);     // size of weights
  EXPECT_EQ(l.ith_parameter(1).size(), 2u);      // size of bias
  EXPECT_EQ(l.parallelize(), true);              // if layer can be parallelized
  EXPECT_STREQ(l.layer_type().c_str(), "conv");  // string with layer type
}

inline void randomize_tensor(tensor_t &tensor) {
  for (auto &vec : tensor) {
    uniform_rand(vec.begin(), vec.end(), -1.0, 1.0);
  }
}

// prepare tensor buffers for unit test
class tensor_buf {
 public:
  tensor_buf(tensor_buf &other)
    : in_data_(other.in_data_), out_data_(other.out_data_) {
    for (auto &d : in_data_) in_ptr_.push_back(&d);
    for (auto &d : out_data_) out_ptr_.push_back(&d);
  }

  explicit tensor_buf(const layer &l, bool randomize = true)
    : in_data_(l.in_channels()),
      out_data_(l.out_channels()),
      in_ptr_(l.in_channels()),
      out_ptr_(l.out_channels()) {
    for (size_t i = 0; i < l.in_channels(); i++) {
      in_data_[i].resize(1, vec_t(l.in_shape()[i].size()));
      in_ptr_[i] = &in_data_[i];
    }

    for (size_t i = 0; i < l.out_channels(); i++) {
      out_data_[i].resize(1, vec_t(l.out_shape()[i].size()));
      out_ptr_[i] = &out_data_[i];
    }

    if (randomize) {
      for (auto &tensor : in_data_) randomize_tensor(tensor);
      for (auto &tensor : out_data_) randomize_tensor(tensor);
    }
  }

  tensor_t &in_at(size_t i) { return in_data_[i]; }
  tensor_t &out_at(size_t i) { return out_data_[i]; }

  std::vector<tensor_t *> &in_buf() { return in_ptr_; }
  std::vector<tensor_t *> &out_buf() { return out_ptr_; }

  friend std::ostream &operator<<(std::ostream &out, const tensor_buf &buffer);

 private:
  std::vector<tensor_t> in_data_, out_data_;
  std::vector<tensor_t *> in_ptr_, out_ptr_;
};

/**
 * Print tensor buffer. May be useful for tests
 * @param out
 * @param buffer
 * @return
 */
std::ostream &operator<<(std::ostream &out, const tensor_buf &buffer) {
  out << "In data:" << std::endl;
  for (auto &ten : buffer.in_data_) {
    out << "Tensor" << std::endl;
    for (auto &vec : ten)
      for (auto &n : vec) out << n << " ";
    out << std::endl << std::endl;
  }
  out << std::endl;
  out << "Out data:" << std::endl;
  for (auto &ten : buffer.out_data_) {
    out << "Tensor" << std::endl;
    for (auto &vec : ten)
      for (auto &n : vec) out << n << " ";
    out << std::endl << std::endl;
  }
  return out;
}

TEST(convolutional, fprop) {
  convolutional_layer l(5, 5, 3, 1, 2);
  l.setup(false);

  // clang-format off
  vec_t in = {
      3,  2, 1, 5, 2,
      3,  0, 2, 0, 1,
      0,  6, 1, 1, 10,
      3, -1, 2, 9, 0,
      1,  2, 1, 5, 5
  };

  vec_t weight = {
      0.3,   0.1,  0.2,
      0.0,  -0.1, -0.1,
      0.05, -0.2, 0.05,

      0.0, -0.1,  0.1,
      0.1, -0.2,  0.3,
      0.2, -0.3,  0.2
  };

  vec_t expected = {
      -0.05, 1.65, 1.45,
       1.05, 0.00, -2.0,
       0.40, 1.15, 0.80,

      -0.80, 1.10, 2.10,
       0.60, 1.50, 0.70,
       0.40, 3.30, -1.0
  };
  // clang-format on

  l.ith_parameter(0).set_data(Tensor<float_t>(weight));
  auto out     = l.forward({{in}});
  vec_t result = (*out[0])[0];

  for (size_t i = 0; i < result.size(); i++) {
    EXPECT_NEAR(expected[i], result[i], 1E-5);
  }
}

/*
TEST(convolutional, with_stride) {
  convolutional_layer l(5, 5, 3, 1, 1, padding::valid, true, 2, 2);
  l.setup(false);

  // clang-format off
  vec_t in = {
      0.0, 1.0, 2.0, 3.0, 4.0,
      1.0, 2.0, 3.0, 4.0, 5.0,
      2.0, 3.0, 4.0, 5.0, 6.0,
      3.0, 4.0, 5.0, 6.0, 7.0,
      4.0, 5.0, 6.0, 7.0, 8.0
  };

  vec_t weight = {
      0.50, 0.50, 0.50,
      0.50, 0.50, 0.50,
      0.50, 0.50, 0.50
  };

  vec_t bias = {
      0.50
  };

  vec_t expected_out = {
      9.50, 18.5,
      18.5, 27.5
  };
  // clang-format on

  l.ith_parameter(0).set_data(Tensor<float_t>(weight));
  l.ith_parameter(1).set_data(Tensor<float_t>(bias));

  auto out         = l.forward({{in}});
  vec_t result_out = (*out[0])[0];

  for (size_t i = 0; i < result_out.size(); i++) {
    EXPECT_NEAR(expected_out[i], result_out[i], 1E-5);
  }

  // clang-format off
  vec_t curr_delta = {
      -1.00, 2.00,
       3.00, 0.00
  };

  vec_t expected_prev_delta = {
      -0.50, -0.50, 0.50, 1.00, 1.00,
      -0.50, -0.50, 0.50, 1.00, 1.00,
       1.00,  1.00, 2.00, 1.00, 1.00,
       1.50,  1.50, 1.50, 0.00, 0.00,
       1.50,  1.50, 1.50, 0.00, 0.00
  };

  vec_t expected_dw = {
       10.0, 14.0, 18.0,
       14.0, 18.0, 22.0,
       18.0, 22.0, 26.0
  };

  vec_t expected_db = {
       4.00
  };
  // clang-format on

  l.ith_parameter(0).clear_grads();
  l.ith_parameter(1).clear_grads();

  vec_t result_prev_delta =
    l.backward(std::vector<tensor_t>{{curr_delta}})[0][0];
  vec_t result_dw = l.ith_parameter(0).grad()->toTensor()[0];
  vec_t result_db = l.ith_parameter(1).grad()->toTensor()[0];

  for (size_t i = 0; i < result_prev_delta.size(); i++) {
    EXPECT_FLOAT_EQ(expected_prev_delta[i], result_prev_delta[i]);
  }

  for (size_t i = 0; i < result_dw.size(); i++) {
    EXPECT_FLOAT_EQ(expected_dw[i], result_dw[i]);
  }

  for (size_t i = 0; i < result_db.size(); i++) {
    EXPECT_FLOAT_EQ(expected_db[i], result_db[i]);
  }
}
*/

// test for AVX backends

#ifdef CNN_USE_AVX
TEST(convolutional, fprop_avx) {
  convolutional_layer l(7, 7, 5, 1, 2);

  tensor_buf buf(l), buf2(l);

  l.set_backend_type(tiny_dnn::core::backend_t::internal);
  l.forward_propagation(buf.in_buf(), buf.out_buf());

  l.set_backend_type(tiny_dnn::core::backend_t::avx);
  l.forward_propagation(buf.in_buf(), buf2.out_buf());

  vec_t &out_avx   = buf2.out_at(0)[0];
  vec_t &out_noavx = buf.out_at(0)[0];

  for (size_t i = 0; i < out_avx.size(); i++) {
    // check if all outputs between default backend and avx backend are the
    // same
    EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
  }
}

TEST(convolutional, bprop_avx) {
  convolutional_layer l(7, 7, 5, 1, 2);

  tensor_buf data(l), grad1(l);
  tensor_buf grad2(grad1);

  l.set_backend_type(tiny_dnn::core::backend_t::internal);

  l.forward_propagation(data.in_buf(), data.out_buf());
  l.back_propagation(data.in_buf(), data.out_buf(), grad1.out_buf(),
                     grad1.in_buf());

  l.set_backend_type(tiny_dnn::core::backend_t::avx);

  l.forward_propagation(data.in_buf(), data.out_buf());
  l.back_propagation(data.in_buf(), data.out_buf(), grad2.out_buf(),
                     grad2.in_buf());

  for (size_t ch = 0; ch < l.out_channels(); ch++) {
    vec_t &out_noavx = grad1.in_at(ch)[0];
    vec_t &out_avx   = grad2.in_at(ch)[0];
    for (size_t i = 0; i < out_avx.size(); i++) {
      EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
    }
  }
}

TEST(convolutional, fprop_avx_1x1out) {
  convolutional_layer l(5, 5, 5, 1, 2);

  tensor_buf buf(l), buf2(l);

  l.set_backend_type(tiny_dnn::core::backend_t::internal);

  l.forward_propagation(buf.in_buf(), buf.out_buf());

  l.set_backend_type(tiny_dnn::core::backend_t::avx);

  l.forward_propagation(buf.in_buf(), buf2.out_buf());

  vec_t &out_avx   = buf2.out_at(0)[0];
  vec_t &out_noavx = buf.out_at(0)[0];

  for (size_t i = 0; i < out_avx.size(); i++) {
    EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
  }
}

TEST(convolutional, bprop_avx_1x1out) {
  convolutional_layer l(5, 5, 5, 1, 2);

  tensor_buf data(l), grad1(l);
  tensor_buf grad2(grad1);

  l.set_backend_type(tiny_dnn::core::backend_t::internal);

  l.forward_propagation(data.in_buf(), data.out_buf());
  l.back_propagation(data.in_buf(), data.out_buf(), grad1.out_buf(),
                     grad1.in_buf());

  l.set_backend_type(tiny_dnn::core::backend_t::avx);

  l.forward_propagation(data.in_buf(), data.out_buf());
  l.back_propagation(data.in_buf(), data.out_buf(), grad2.out_buf(),
                     grad2.in_buf());

  for (size_t ch = 0; ch < l.out_channels(); ch++) {
    vec_t &out_noavx = grad1.in_at(ch)[0];
    vec_t &out_avx   = grad2.in_at(ch)[0];
    for (size_t i = 0; i < out_avx.size(); i++) {
      EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
    }
  }
}

TEST(convolutional, fprop_avx_hstride) {
  convolutional_layer l(7, 7, 5, 1, 2, padding::valid, true, 1, 2);

  tensor_buf buf(l), buf2(l);

  l.set_backend_type(tiny_dnn::core::backend_t::internal);

  l.forward_propagation(buf.in_buf(), buf.out_buf());

  l.set_backend_type(tiny_dnn::core::backend_t::avx);

  l.forward_propagation(buf.in_buf(), buf2.out_buf());

  vec_t &out_avx   = buf2.out_at(0)[0];
  vec_t &out_noavx = buf.out_at(0)[0];

  for (size_t i = 0; i < out_avx.size(); i++) {
    EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
  }
}

TEST(convolutional, bprop_avx_hstride) {
  convolutional_layer l(7, 7, 5, 1, 2, padding::valid, true, 1, 2);

  tensor_buf data(l), grad1(l);
  tensor_buf grad2(grad1);

  l.set_backend_type(tiny_dnn::core::backend_t::internal);

  l.forward_propagation(data.in_buf(), data.out_buf());
  l.back_propagation(data.in_buf(), data.out_buf(), grad1.out_buf(),
                     grad1.in_buf());

  l.set_backend_type(tiny_dnn::core::backend_t::avx);

  l.forward_propagation(data.in_buf(), data.out_buf());
  l.back_propagation(data.in_buf(), data.out_buf(), grad2.out_buf(),
                     grad2.in_buf());

  for (size_t ch = 0; ch < l.out_channels(); ch++) {
    vec_t &out_noavx = grad1.in_at(ch)[0];
    vec_t &out_avx   = grad2.in_at(ch)[0];
    for (size_t i = 0; i < out_avx.size(); i++) {
      EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
    }
  }
}

TEST(convolutional, fprop_avx_hstride_1x1out) {
  convolutional_layer l(5, 5, 5, 1, 2, padding::valid, true, 1, 2);

  tensor_buf buf(l), buf2(l);

  l.set_backend_type(tiny_dnn::core::backend_t::internal);

  l.forward_propagation(buf.in_buf(), buf.out_buf());

  l.set_backend_type(tiny_dnn::core::backend_t::avx);

  l.forward_propagation(buf.in_buf(), buf2.out_buf());

  vec_t &out_avx   = buf2.out_at(0)[0];
  vec_t &out_noavx = buf.out_at(0)[0];

  for (size_t i = 0; i < out_avx.size(); i++) {
    EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
  }
}

TEST(convolutional, bprop_avx_hstride_1x1out) {
  convolutional_layer l(5, 5, 5, 1, 2, padding::valid, true, 1, 2);

  tensor_buf data(l), grad1(l);
  tensor_buf grad2(grad1);

  l.set_backend_type(tiny_dnn::core::backend_t::internal);

  l.forward_propagation(data.in_buf(), data.out_buf());
  l.back_propagation(data.in_buf(), data.out_buf(), grad1.out_buf(),
                     grad1.in_buf());

  l.set_backend_type(tiny_dnn::core::backend_t::avx);

  l.forward_propagation(data.in_buf(), data.out_buf());
  l.back_propagation(data.in_buf(), data.out_buf(), grad2.out_buf(),
                     grad2.in_buf());

  for (size_t ch = 0; ch < l.out_channels(); ch++) {
    vec_t &out_noavx = grad1.in_at(ch)[0];
    vec_t &out_avx   = grad2.in_at(ch)[0];
    for (size_t i = 0; i < out_avx.size(); i++) {
      EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
    }
  }
}

TEST(convolutional, fprop_avx_wstride) {
  convolutional_layer l(7, 7, 5, 1, 2, padding::valid, true, 2, 1);

  tensor_buf buf(l), buf2(l);

  l.set_backend_type(tiny_dnn::core::backend_t::internal);

  l.forward_propagation(buf.in_buf(), buf.out_buf());

  l.set_backend_type(tiny_dnn::core::backend_t::avx);

  l.forward_propagation(buf.in_buf(), buf2.out_buf());

  vec_t &out_avx   = buf2.out_at(0)[0];
  vec_t &out_noavx = buf.out_at(0)[0];

  for (size_t i = 0; i < out_avx.size(); i++) {
    EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
  }
}

TEST(convolutional, bprop_avx_wstride) {
  convolutional_layer l(7, 7, 5, 1, 2, padding::valid, true, 2, 1);

  tensor_buf data(l), grad1(l);
  tensor_buf grad2(grad1);

  l.set_backend_type(tiny_dnn::core::backend_t::internal);

  l.forward_propagation(data.in_buf(), data.out_buf());
  l.back_propagation(data.in_buf(), data.out_buf(), grad1.out_buf(),
                     grad1.in_buf());

  l.set_backend_type(tiny_dnn::core::backend_t::avx);

  l.forward_propagation(data.in_buf(), data.out_buf());
  l.back_propagation(data.in_buf(), data.out_buf(), grad2.out_buf(),
                     grad2.in_buf());

  for (size_t ch = 0; ch < l.out_channels(); ch++) {
    vec_t &out_noavx = grad1.in_at(ch)[0];
    vec_t &out_avx   = grad2.in_at(ch)[0];
    for (size_t i = 0; i < out_avx.size(); i++) {
      EXPECT_NEAR(out_avx[i], out_noavx[i], 1E-5);
    }
  }
}

#endif  // CNN_USE_AVX

#ifdef CNN_USE_NNPACK
TEST(convolutional, fprop_nnp) {
  convolutional_layer l(5, 5, 3, 1, 2, padding::valid, true, 1, 1,
                        core::backend_t::nnpack);

  // layer::forward_propagation expects tensors, even if we feed only one
  // input at a time
  auto create_simple_tensor = [](size_t vector_size) {
    return tensor_t(1, vec_t(vector_size));
  };

  // create simple tensors that wrap the payload vectors of the correct size
  tensor_t in_tensor     = create_simple_tensor(25),
           out_tensor    = create_simple_tensor(18),
           a_tensor      = create_simple_tensor(18),
           weight_tensor = create_simple_tensor(18),
           bias_tensor   = create_simple_tensor(2);

  tensor_buf buf(l, false);

  ASSERT_EQ(l.in_shape()[1].size(), size_t(18));  // weight
  // short-hand references to the payload vectors
  vec_t &in = buf.in_at(0)[0], &weight = buf.in_at(1)[0];
  uniform_rand(in.begin(), in.end(), -1.0, 1.0);

  l.setup(false);
  {
    l.forward_propagation(buf.in_buf(), buf.out_buf());

    vec_t &out = buf.out_at(0)[0];

    for (auto o : out) EXPECT_DOUBLE_EQ(o, tiny_dnn::float_t(0.0));
  }

  // clang-format off
  weight[0] = 0.3;  weight[1] = 0.1;   weight[2] = 0.2;
  weight[3] = 0.0;  weight[4] = -0.1;  weight[5] = -0.1;
  weight[6] = 0.05; weight[7] = -0.2;  weight[8] = 0.05;

  weight[9]  = 0.0; weight[10] = -0.1; weight[11] = 0.1;
  weight[12] = 0.1; weight[13] = -0.2; weight[14] = 0.3;
  weight[15] = 0.2; weight[16] = -0.3; weight[17] = 0.2;

  in[0]  = 3; in[1]  = 2;  in[2]  = 1; in[3]  = 5; in[4]  = 2;
  in[5]  = 3; in[6]  = 0;  in[7]  = 2; in[8]  = 0; in[9]  = 1;
  in[10] = 0; in[11] = 6;  in[12] = 1; in[13] = 1; in[14] = 10;
  in[15] = 3; in[16] = -1; in[17] = 2; in[18] = 9; in[19] = 0;
  in[20] = 1; in[21] = 2;  in[22] = 1; in[23] = 5; in[24] = 5;
  // clang-format on

  {
    l.forward_propagation(buf.in_buf(), buf.out_buf());

    vec_t &out = buf.out_at(0)[0];

    EXPECT_NEAR(float_t(-0.05), out[0], 1E-5);
    EXPECT_NEAR(float_t(1.65), out[1], 1E-5);
    EXPECT_NEAR(float_t(1.45), out[2], 1E-5);
    EXPECT_NEAR(float_t(1.05), out[3], 1E-5);
    EXPECT_NEAR(float_t(0.00), out[4], 1E-5);
    EXPECT_NEAR(float_t(-2.0), out[5], 1E-5);
    EXPECT_NEAR(float_t(0.40), out[6], 1E-5);
    EXPECT_NEAR(float_t(1.15), out[7], 1E-5);
    EXPECT_NEAR(float_t(0.80), out[8], 1E-5);
  }
}
#endif  // CNN_USE_NNPACK

TEST(convolutional, gradient_check) {  // tanh - mse
  network<sequential> nn;
  nn << convolutional_layer(5, 5, 3, 1, 1) << activation::tanh();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check2) {  // sigmoid - mse
  network<sequential> nn;
  nn << convolutional_layer(5, 5, 3, 1, 1) << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check3) {  // rectified - mse
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1) << relu();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check4) {  // identity - mse
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1);

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check5) {  // sigmoid - cross-entropy
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1) << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<cross_entropy>(
    test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check6) {  // sigmoid - absolute
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1) << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<absolute>(test_data.first, test_data.second,
                                          epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check7) {  // sigmoid - absolute eps
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1) << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<absolute_eps<100>>(
    test_data.first, test_data.second, epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check8_pad_same) {  // sigmoid - mse - padding same
  network<sequential> nn;

  nn << convolutional_layer(5, 5, 3, 1, 1, padding::same, true, 1, 1,
                            core::backend_t::internal)
     << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional, gradient_check9_w_stride) {  // sigmoid - mse - w_stride > 1
  network<sequential> nn;

  nn << convolutional_layer(3, 3, 1, 1, 1, padding::valid, true, 2, 1,
                            core::backend_t::internal)
     << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size(), 1);
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional,
     gradient_check10_h_stride) {  // sigmoid - mse - h_stride > 1
  network<sequential> nn;

  nn << convolutional_layer(3, 3, 1, 1, 1, padding::valid, true, 1, 2,
                            core::backend_t::internal)
     << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size(), 1);
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional,
     gradient_check11_connection_tbl) {  // sigmoid - mse - has connection-tbl
  network<sequential> nn;
  bool tbl[3 * 3] = {true, false, true, false, true, false, true, true, false};

  core::connection_table connections(tbl, 3, 3);

  nn << convolutional_layer(7, 7, 3, 3, 1, connections, padding::valid, true, 1,
                            1, core::backend_t::internal)
     << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(convolutional,
     gradient_check12_pad_same) {  // sigmoid - mse - padding same
  network<sequential> nn;

  nn << fully_connected_layer(10, 5 * 5)
     << convolutional_layer(5, 5, 3, 1, 1, padding::same, true, 1, 1,
                            core::backend_t::internal)
     << sigmoid();

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

/* todo (karandesai) : deal with serialization after parameter integration later
 * uncomment after fixing
TEST(convolutional, read_write) {
  convolutional_layer l1(5, 5, 3, 1, 1);
  convolutional_layer l2(5, 5, 3, 1, 1);

  l1.init_weight();
  l2.init_weight();

  serialization_test(l1, l2);
}

TEST(convolutional, read_write2) {
#define O true
#define X false
  static const bool connection[] = {O, X, X, X, O, O, O, O, X,
                                    X, X, O, O, O, O, X, X, X};
#undef O
#undef X
  convolutional_layer layer1(14, 14, 5, 3, 6,
                             core::connection_table(connection, 3, 6));
  convolutional_layer layer2(14, 14, 5, 3, 6,
                             core::connection_table(connection, 3, 6));
  layer1.init_weight();
  layer2.init_weight();

  serialization_test(layer1, layer2);
}
*/

TEST(convolutional, copy_and_pad_input_same) {
  core::conv_params params;
  params.in        = shape3d(5, 5, 1);
  params.weight    = shape3d(3, 3, 2);
  params.in_padded = shape3d(7, 7, 1);
  params.out       = shape3d(3, 3, 1);
  params.pad_type  = padding::same;  // test target
  params.w_stride  = 1;
  params.h_stride  = 1;

  core::Conv2dPadding conv2d_padding(params);

  auto create_tensor = [](size_t batch_size, size_t vector_size) {
    return tensor_t(batch_size, vec_t(vector_size));
  };

  tensor_t in_tensor = create_tensor(1, 1 * 5 * 5), out_tensor;

  fill_tensor(in_tensor, float_t(1));

  /* @in_tensor   --->   @out_tensor
   *
   *    1 1 1             0 0 0 0 0
   *    1 1 1             0 1 1 1 0
   *    1 1 1             0 1 1 1 0
   *                      0 1 1 1 0
   *                      0 0 0 0 0
   */

  conv2d_padding.copy_and_pad_input(in_tensor, out_tensor);

  EXPECT_EQ(out_tensor[0][7], float_t(0));
  EXPECT_EQ(out_tensor[0][8], float_t(1));
  EXPECT_EQ(out_tensor[0][9], float_t(1));
  EXPECT_EQ(out_tensor[0][10], float_t(1));
  EXPECT_EQ(out_tensor[0][11], float_t(1));
  EXPECT_EQ(out_tensor[0][12], float_t(1));
  EXPECT_EQ(out_tensor[0][13], float_t(0));
}

TEST(convolutional, copy_and_unpad_delta_same) {
  core::conv_params params;
  params.in        = shape3d(3, 3, 1);
  params.weight    = shape3d(2, 2, 1);
  params.in_padded = shape3d(5, 5, 1);
  params.out       = shape3d(2, 2, 1);
  params.pad_type  = padding::same;  // test target
  params.w_stride  = 1;
  params.h_stride  = 1;

  core::Conv2dPadding conv2d_padding(params);

  auto create_tensor = [](size_t batch_size, size_t vector_size) {
    return tensor_t(batch_size, vec_t(vector_size));
  };

  tensor_t in_tensor = create_tensor(1, 1 * 5 * 5), out_tensor;

  fill_tensor(in_tensor, float_t(0));

  /*
   * @in_tensor   --->   @out_tensor
   *
   * 0 0 0 0 0             1 1 1
   * 0 1 1 1 0             1 1 1
   * 0 1 1 1 0             1 1 1
   * 0 1 1 1 0
   * 0 0 0 0 0
   *
   */

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      in_tensor[0][6 + 5 * i + j] = float_t(1);
    }
  }

  conv2d_padding.copy_and_unpad_delta(in_tensor, out_tensor);

  for (size_t i = 0; i < out_tensor[0].size(); ++i) {
    EXPECT_EQ(out_tensor[0][i], float_t(1));
  }
}

}  // namespace tiny_dnn
