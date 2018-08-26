/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <vector>

#include "tiny_dnn/core/framework/tensor_utils.h"

namespace tiny_dnn {

TEST(tensor, constructors) {
  Tensor<float_t> t1;
  Tensor<float_t> t2({2, 2, 2, 2});
  t2.fill(float_t(2.0));

  t1 = t2;  // invoke assign copy ctor

  // check that t2 values has been copyied to t1
  for (auto it = t1.host_begin(); it != t1.host_end(); ++it) {
    EXPECT_EQ(*it, float_t(2.0));
  }

  t1 = Tensor<float_t>({2, 2, 3, 4, 2}, 0);  // invoke copy ctor

  // check that t1 have default values
  for (auto it = t1.host_begin(); it != t1.host_end(); ++it) {
    EXPECT_EQ(*it, float_t(0.0));
  }

  t2 = Tensor<float_t>({1, 1, 1, 1, 2}, static_cast<float_t>(1.1));
  EXPECT_EQ(*t2.host_begin(), float_t(1.1));
}

TEST(tensor, shape) {
  Tensor<float_t> tensor({1, 2, 2, 2});

  EXPECT_EQ(tensor.shape()[0], size_t(1));
  EXPECT_EQ(tensor.shape()[1], size_t(2));
  EXPECT_EQ(tensor.shape()[2], size_t(2));
  EXPECT_EQ(tensor.shape()[3], size_t(2));
}

TEST(tensor, size) {
  Tensor<float_t> tensor({2, 2, 2, 2});

  EXPECT_EQ(tensor.size(), size_t(2 * 2 * 2 * 2));
}

TEST(tensor, view1) {
  Tensor<float_t> tensor({3, 3, 3, 3}, 2);
  auto t_view = tensor.subView({0, 2}, {0, 1}, {0, 3}, {0, 3});
  EXPECT_EQ(t_view.shape()[0], size_t(2));
  EXPECT_EQ(t_view.shape()[1], size_t(1));
  EXPECT_EQ(t_view.shape()[2], size_t(3));
  EXPECT_EQ(t_view.shape()[3], size_t(3));
  t_view.host_at(0, 0, 0, 0) = -1;
  t_view.host_at(1, 0, 1, 1) = 3;
  EXPECT_EQ(tensor.host_at(0, 0, 0, 0), -1);
}

TEST(tensor, reshape) {
  Tensor<float_t> tensor({1, 2, 2, 2}, 2);
  tensor.reshape({4, 1, 2});
  EXPECT_EQ(tensor.shape()[0], size_t(4));
  EXPECT_EQ(tensor.shape().size(), size_t(3));
  EXPECT_EQ(tensor.host_at(2, 0, 1), size_t(2));
}

TEST(tensor, fill) {
  Tensor<float_t> tensor({2, 2, 2, 2});

  // fill all tensor values with ones

  tensor.fill(float_t(1.0));

  for (auto it = tensor.host_begin(); it != tensor.host_end(); ++it) {
    EXPECT_EQ(*it, float_t(1.0));
  }

  // fill all tensor values with twos

  tensor.fill(float_t(2.0));

  for (auto it = tensor.host_begin(); it != tensor.host_end(); ++it) {
    EXPECT_EQ(*it, float_t(2.0));
  }
}

TEST(tensor, add1) {
  Tensor<float_t> t1({2, 2, 2, 2});
  Tensor<float_t> t2({2, 2, 2, 2});

  // fill tensor with initial values

  t1.fill(float_t(1.0));
  t2.fill(float_t(3.0));

  // compute element-wise sum along all tensor values

  Tensor<float_t> t3;

  layer_add(t3, t1, t2);

  // check that sum is okay

  for (auto it = t3.host_begin(); it != t3.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(4.0), 1e-5);
  }
}

TEST(tensor, add2a) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(1.0));

  // compute element-wise sum along all tensor values

  Tensor<float_t> t2;

  // check that sum is okay

  layer_add(t2, float_t(2.0), t);

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(3.0), 1e-5);
  }
}

TEST(tensor, add2b) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(1.0));

  // compute element-wise sum along all tensor values

  Tensor<float_t> t2;

  // check that sum is okay

  layer_add(t2, t, float_t(2.0));

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(3.0), 1e-5);
  }
}

TEST(tensor, add3) {
  Tensor<float_t> t1({2, 2, 2, 2});
  Tensor<float_t> t2({4, 4, 4, 4});

  // compute element-wise sum along all tensor values.
  // Expect a throw since shapes are different

  Tensor<float_t> t3;

  EXPECT_THROW(layer_add(t3, t1, t2), nn_error);
}

TEST(tensor, add4) {
  Tensor<float_t> t1({2, 2, 2, 2}, 5);
  Tensor<float_t> t2({4, 4, 4, 4}, 3);
  auto t3 = t2.subView({0, 2}, {0, 2}, {0, 2}, {0, 2});

  // compute element-wise sum along all tensor values.

  Tensor<float_t> t4;

  // check that sum is okay

  layer_add(t4, t1, t3);

  for (auto it = t4.host_begin(); it != t4.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(8.0), 1e-5);
  }
}

TEST(tensor, sub1) {
  Tensor<float_t> t1({2, 2, 2, 2});
  Tensor<float_t> t2({2, 2, 2, 2});

  // fill tensor with initial values

  t1.fill(float_t(1.0));
  t2.fill(float_t(3.0));

  // compute element-wise subtraction along all tensor values

  Tensor<float_t> t3;
  layer_sub(t3, t1, t2);

  // check that difference is okay

  for (auto it = t3.host_begin(); it != t3.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(-2.0), 1e-5);
  }
}

TEST(tensor, sub2a) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(1.0));

  // compute element-wise subtraction along all tensor values

  Tensor<float_t> t2;

  layer_sub(t2, t, float_t(2.0));

  // check that subtraction is okay

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(-1.0), 1e-5);
  }
}

TEST(tensor, sub2b) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(2.0));

  // compute element-wise subtraction along all tensor values

  Tensor<float_t> t2;

  layer_sub(t2, float_t(1.0), t);

  // check that subtraction is okay

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(-1.0), 1e-5);
  }
}

TEST(tensor, sub3) {
  Tensor<float_t> t1({2, 2, 2, 2});
  Tensor<float_t> t2({4, 4, 4, 4});

  // compute element-wise subtraction along all tensor values.
  // Expect a throw since shapes are different

  Tensor<float_t> t3;

  EXPECT_THROW(layer_sub(t3, t1, t2), nn_error);
}
TEST(tensor, sub4) {
  Tensor<float_t> t1({2, 2, 2, 2}, 5);
  Tensor<float_t> t2({4, 4, 4, 4}, 3);
  auto t3 = t2.subView({0, 2}, {0, 2}, {0, 2}, {0, 2});

  // compute element-wise sum along all tensor values.

  Tensor<float_t> t4;

  // check that diffrence is okay

  layer_sub(t4, t1, t3);

  for (auto it = t4.host_begin(); it != t4.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(2.0), 1e-5);
  }
}
TEST(tensor, mul1) {
  Tensor<float_t> t1({2, 2, 2, 2});
  Tensor<float_t> t2({2, 2, 2, 2});

  // fill tensor with initial values

  t1.fill(float_t(2.0));
  t2.fill(float_t(3.0));

  // compute element-wise multiplication along all tensor values

  Tensor<float_t> t3;

  layer_mul(t3, t1, t2);

  // check that product is okay

  for (auto it = t3.host_begin(); it != t3.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(6.0), 1e-5);
  }
}

TEST(tensor, mul2a) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(2.0));

  // compute element-wise multiplication along all tensor values

  Tensor<float_t> t2;

  layer_mul(t2, t, float_t(2.0));

  // check that multiplication is okay

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(4.0), 1e-5);
  }
}

TEST(tensor, mul2b) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(2.0));

  // compute element-wise multiplication along all tensor values

  Tensor<float_t> t2;

  layer_mul(t2, float_t(2.0), t);

  // check that product is okay

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(4.0), 1e-5);
  }
}

TEST(tensor, mul3) {
  Tensor<float_t> t1({2, 2, 2, 2});
  Tensor<float_t> t2({4, 4, 4, 4});

  // compute element-wise multiplication along all tensor values.
  // Expect a throw since shapes are different

  Tensor<float_t> t3;

  EXPECT_THROW(layer_mul(t3, t1, t2), nn_error);
}

TEST(tensor, mul4) {
  Tensor<float_t> t1({2, 2, 2, 2}, 5);
  Tensor<float_t> t2({4, 4, 4, 4}, 3);
  auto t3 = t2.subView({0, 2}, {0, 2}, {0, 2}, {0, 2});

  // compute element-wise sum along all tensor values.

  Tensor<float_t> t4;

  layer_mul(t4, t1, t3);

  // check that product is okay
  for (auto it = t4.host_begin(); it != t4.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(15.0), 1e-5);
  }
}

TEST(tensor, div1) {
  Tensor<float_t> t1({2, 2, 2, 2});
  Tensor<float_t> t2({2, 2, 2, 2});

  // fill tensor with initial values

  t1.fill(float_t(1.0));
  t2.fill(float_t(2.0));

  // compute element-wise division along all tensor values

  Tensor<float_t> t3;

  layer_div(t3, t1, t2);

  // check that division is okay

  for (auto it = t3.host_begin(); it != t3.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(0.5), 1e-5);
  }
}

TEST(tensor, div2a) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(1.0));

  // compute element-wise division along all tensor values

  Tensor<float_t> t2;

  layer_div(t2, t, float_t(2.0));

  // check that division is okay

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(0.5), 1e-5);
  }
}

TEST(tensor, div2b) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(2.0));

  // compute element-wise division along all tensor values

  Tensor<float_t> t2;

  layer_div(t2, float_t(1.0), t);

  // check that division is okay

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(0.5), 1e-5);
  }
}

TEST(tensor, div3) {
  Tensor<float_t> t1({2, 2, 2, 2});
  Tensor<float_t> t2({4, 4, 4, 4});

  // compute element-wise division along all tensor values.
  // Expect a throw since shapes are different

  Tensor<float_t> t3;

  EXPECT_THROW(layer_div(t3, t1, t2), nn_error);
}

TEST(tensor, div4) {
  Tensor<float_t> t1({2, 2, 2, 2});
  Tensor<float_t> t2({2, 2, 2, 2});

  // fill tensor with initial values

  t1.fill(float_t(1.0));
  t2.fill(float_t(0.0));

  // compute element-wise division along all tensor values

  Tensor<float_t> t3;

  layer_div(t3, t1, t2);

  // check that division is NaN

  for (auto it = t3.host_begin(); it != t3.host_end(); ++it) {
    EXPECT_TRUE(std::isnan(*it));
  }
}

TEST(tensor, div5) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(1.0));

  // compute element-wise division along all tensor values

  Tensor<float_t> t2;

  layer_div(t2, t, float_t(0.0));

  // check that division is NaN

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_TRUE(std::isnan(*it));
  }
}

TEST(tensor, div6) {
  Tensor<float_t> t1({2, 2, 2, 2}, 5);
  Tensor<float_t> t2({4, 4, 4, 4}, 2);
  auto t3 = t2.subView({0, 2}, {0, 2}, {0, 2}, {0, 2});

  // compute element-wise sum along all tensor values.

  Tensor<float_t> t4;

  // check that result is okay

  layer_div(t4, t1, t3);

  for (auto it = t4.host_begin(); it != t4.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(2.5), 1e-5);
  }
}

TEST(tensor, sqrt1) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values
  t.fill(float_t(4.0));

  // compute element-wise square root along all tensor values

  Tensor<float_t> t2;

  layer_sqrt(t2, t);

  // check that root is okay

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_NEAR(*it, float_t(2.0), 1e-5);
  }
}

TEST(tensor, sqrt2) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values
  t.fill(float_t(-1.0));

  // compute element-wise square root along all tensor values

  Tensor<float_t> t2;

  layer_sqrt(t2, t);

  // check that division is NaN

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_TRUE(std::isnan(*it));
  }
}

TEST(tensor, exp) {
  Tensor<float_t> t({2, 2, 2, 2});

  // fill tensor with initial values
  t.fill(float_t(-1.0));

  // compute element-wise exponent along all tensor values

  Tensor<float_t> t2;

  layer_exp(t2, t);

  // check that exponent calculated right

  for (auto it = t2.host_begin(); it != t2.host_end(); ++it) {
    EXPECT_NEAR(*it, std::exp(float_t(-1.0)), 1e-5);
  }
}

TEST(tensor, dim) {
  Tensor<float_t> t({2, 2, 2, 2});
  EXPECT_EQ(t.dim(), 4u);
  Tensor<float_t> t2({4, 2});
  EXPECT_EQ(t2.dim(), 2u);
  auto t3 = t2.subView({0, 2}, {0, 1});
  EXPECT_EQ(t3.dim(), 2u);
}

template <size_t N>
std::ostream &print_tester(std::ostream &os) {
  os << "\nPrinting " << N << "-dimensional Tensor"
     << ":\n\n";
  std::vector<size_t> shape(N, 2);
  shape.back() = 3;

  Tensor<float_t> t(shape);
  t.fill(float_t{1.0});
  os << t;
  print_tester<N - 1>(os);
  return os;
}

template <>
std::ostream &print_tester<0>(std::ostream &os) {
  return os;
}

TEST(tensor, print) { print_tester<5>(std::cout); }

TEST(tensor, print_view) {
  Tensor<float_t> tensor({3, 3, 3, 3}, 2);
  auto t_view = tensor.subView({0, 2}, {0, 1}, {0, 3}, {0, 3});
  t_view.host_at(0, 0, 0, 0) = -1;
  t_view.host_at(1, 0, 1, 1) = 3;
  std::cout << tensor << std::endl << t_view << std::endl;
}

}  // namespace tiny_dnn
