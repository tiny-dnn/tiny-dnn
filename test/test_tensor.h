/*
    COPYRIGHT

    All contributions by Taiga Nomi
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    All other contributions:
    Copyright (c) 2013-2016, the respective contributors.
    All rights reserved.

    Each contributor holds copyright over their respective contributions.
    The project versioning (Git) records all such contribution source
   information.

    LICENSE

    The BSD 3-Clause License


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
   this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of tiny-dnn nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
   ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
   LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
   USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"

#include "tiny_dnn/core/framework/tensor_utils.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

namespace tiny_dnn {

TEST(tensor, constructors) {
  Tensor<float_t> t1;
  Tensor<float_t> t2({2, 2, 2, 2});
  t2.fill(float_t(2.0));

  t1 = t2;  // invoke assign copy ctor

  // check that t2 values has been copyied to t1
  for (size_t i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(t1.host_data()[i], float_t(2.0));
  }

  t1 = Tensor<float_t>({1, 1, 1, 1});  // invoke copy ctor

  // check that t1 have default values
  for (size_t i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(t1.host_data()[i], float_t(0.0));
  }

  // TODO
  // invoke move assign cto
  // t1 = std::move(t2);

  // check that we moved data
  // EXPECT_EQ(t1.size(), serial_size_t(16));

  // expecting something here is wrong. t2 is in a valid but undefined state,
  // no
  // need to test it.

  // invoke move ctor
  // Tensor<float_t> t3(std::move(t1));

  // check that we moved data
  // EXPECT_EQ(t3.size(), serial_size_t(16));
}

TEST(tensor, shape) {
  Tensor<float_t, 4> tensor({1, 2, 2, 2});

  EXPECT_EQ(tensor.shape()[0], size_t(1));
  EXPECT_EQ(tensor.shape()[1], size_t(2));
  EXPECT_EQ(tensor.shape()[2], size_t(2));
  EXPECT_EQ(tensor.shape()[3], size_t(2));
}

TEST(tensor, size) {
  Tensor<float_t, 4> tensor({2, 2, 2, 2});

  EXPECT_EQ(tensor.size(), size_t(2 * 2 * 2 * 2));
}

TEST(tensor, check_bounds) {
  Tensor<float_t, 4> tensor({1, 2, 2, 1});

  // check bounds with .at() accessor

  EXPECT_NO_THROW(tensor.host_at(0, 0, 0, 0));
  EXPECT_NO_THROW(tensor.host_at(0, 0, 1, 0));
  EXPECT_NO_THROW(tensor.host_at(0, 1, 0, 0));
  EXPECT_NO_THROW(tensor.host_at(0, 1, 1, 0));

  EXPECT_THROW(tensor.host_at(0, 0, 0, 1), nn_error);
  EXPECT_THROW(tensor.host_at(1, 0, 0, 0), nn_error);
  EXPECT_THROW(tensor.host_at(1, 0, 0, 1), nn_error);

  // check bounds with .ptr() accessor

  EXPECT_NO_THROW(tensor.host_ptr(0, 0, 0, 0));
  EXPECT_NO_THROW(tensor.host_ptr(0, 0, 1, 0));
  EXPECT_NO_THROW(tensor.host_ptr(0, 1, 0, 0));
  EXPECT_NO_THROW(tensor.host_ptr(0, 1, 1, 0));

  EXPECT_THROW(tensor.host_ptr(0, 0, 0, 1), nn_error);
  EXPECT_THROW(tensor.host_ptr(1, 0, 0, 0), nn_error);
  EXPECT_THROW(tensor.host_ptr(1, 0, 0, 1), nn_error);
}

TEST(tensor, view1) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});
  Tensor<float_t, 4> t2 = t1.subView({2, 4});
  Tensor<float_t, 4> t3 = t1.subView({0}, {2, 4});

  EXPECT_TRUE(!t1.isSubView());
  EXPECT_TRUE(t2.isSubView());
  EXPECT_TRUE(t3.isSubView());

  EXPECT_EQ(t1.size(), size_t(2 * 2 * 2 * 2));
  EXPECT_EQ(t2.size(), size_t(2 * 4));
  EXPECT_EQ(t3.size(), size_t(2 * 4));
}

TEST(tensor, view2) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});

  EXPECT_NO_THROW(t1.subView({2, 4, 0, 0}));
  EXPECT_NO_THROW(t1.subView({0, 0, 0, 0}, {2, 4, 0, 0}));

  // test that num parameters exceed to num of dimensions
  EXPECT_THROW(t1.subView({2, 4, 0, 0, 0}), nn_error);
  EXPECT_THROW(t1.subView({0, 0, 0, 0}, {2, 4, 0, 0, 0}), nn_error);
  EXPECT_THROW(t1.subView({2, 4, 0, 0, 0}, {1, 2, 0, 0}), nn_error);
  EXPECT_THROW(t1.subView({2, 4, 0, 0, 0}, {1, 2, 0, 0, 0}), nn_error);
}

TEST(tensor, view3) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});

  // test that cannot generate view bigger than current tensor
  EXPECT_THROW(t1.subView({4, 5}), nn_error);
  EXPECT_THROW(t1.subView({1}, {4, 4}), nn_error);
}

TEST(tensor, view4) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});
  Tensor<float_t, 4> t2 = t1.subView({2, 4});

  // modify sub view tensor
  for (size_t i = 0; i < t2.size(); ++i) {
    t2.host_data()[i] = float_t(i + 1);
  }

  // check that root tensor has also been modified
  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_EQ(t1.host_data()[i], float_t(i + 1));
    EXPECT_EQ(t2.host_data()[i], float_t(i + 1));
  }
}

TEST(tensor, view5) {
  Tensor<float_t, 2> t1({3, 3});

  // we create a sub view tensor with size @2x2.
  // Ideally it should represent the top-left matrix.
  // However, since we assume continous memory, it will
  // mask the first four elements from the root view.
  Tensor<float_t, 2> t2 = t1.subView({0}, {2, 2});

  // modify sub view tensor
  for (size_t i = 0; i < t2.size(); ++i) {
    t2.host_data()[i] = float_t(i + 1);
  }

  // check that root tensor has been modified assuming
  // continous memory. The ideal case would be that the
  // new sub ºview can handle non continous memory pointing
  // to the top-left matrix from the root tensor.
  EXPECT_EQ(t1.host_at(0, 0), float_t(1.0));
  EXPECT_EQ(t1.host_at(0, 1), float_t(2.0));
  EXPECT_EQ(t1.host_at(0, 2), float_t(3.0));
  EXPECT_EQ(t1.host_at(1, 0), float_t(4.0));

  // check that the new sub view does not assume the ideal
  // case with non continuous memory.
  EXPECT_TRUE(t1.host_at(0, 0) == float_t(1.0));
  EXPECT_TRUE(t1.host_at(0, 1) == float_t(2.0));
  EXPECT_FALSE(t1.host_at(1, 0) == float_t(3.0));
  EXPECT_FALSE(t1.host_at(1, 1) == float_t(4.0));
}

TEST(tensor, access_data1) {
  Tensor<float_t, 4> tensor({1, 2, 2, 1});

  const std::array<size_t, 4> &shape = tensor.shape();

  for (serial_size_t n = 0; n < shape[0]; ++n) {
    for (serial_size_t w = 0; w < shape[1]; ++w) {
      for (serial_size_t h = 0; h < shape[2]; ++h) {
        for (serial_size_t d = 0; d < shape[3]; ++d) {
          EXPECT_NEAR(tensor.host_at(n, w, h, d), float_t(0.0), 1e-5);
          EXPECT_NEAR(*tensor.host_ptr(n, w, h, d), float_t(0.0), 1e-5);
        }
      }
    }
  }
}

TEST(tensor, access_data2) {
  Tensor<float_t, 4> tensor({1, 2, 2, 1});

  for (size_t i = 0; i < tensor.size(); ++i) {
    EXPECT_EQ(tensor.host_data()[i], float_t(0.0));
  }
}

TEST(tensor, access_data3) {
  Tensor<float_t, 4> tensor({1, 2, 2, 2});

  // modify data using .ptr() accessor

  float_t *ptr1 = tensor.host_ptr(0, 0, 0, 0);
  float_t *ptr2 = tensor.host_ptr(0, 1, 0, 0);

  for (size_t i = 0; i < 8; ++i) {
    ptr1[i] = float_t(i);
  }

  for (size_t i = 0; i < 4; ++i) {
    ptr2[i] = float_t(2.0);
  }

  // check data using .ptr() accessor

  const float_t *ptr11 = tensor.host_ptr(0, 0, 0, 0);
  const float_t *ptr22 = tensor.host_ptr(0, 1, 0, 0);

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(ptr11[i], float_t(i));
  }

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(ptr22[i], float_t(2.0));
  }
}

TEST(tensor, access_data4) {
  Tensor<float_t> tensor({1, 2, 2, 2});

  // modify data using .ptr() accessor

  float_t *ptr1 = tensor.host_ptr(0, 0, 0, 0);
  float_t *ptr2 = tensor.host_ptr(0, 1, 0, 0);

  for (size_t i = 0; i < 4; ++i) {
    ptr1[i] = float_t(i * 1.0);
  }

  for (size_t i = 0; i < 4; ++i) {
    ptr2[i] = float_t(i * 3.0);
  }

  // check data using .at() accessor
  const std::array<float_t, 4> vals1 = {0, 2, 0, 6}, vals2 = {1, 3, 3, 9};
  for (size_t i = 0; i < 2; ++i) {
    for (serial_size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(tensor.host_at(0, i, j, 0), vals1[i * 2 + j]);
    }
  }

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(tensor.host_at(0, i, j, 1), vals2[i * 2 + j]);
    }
  }
}

TEST(tensor, access_data5) {
  Tensor<float_t> tensor({1, 2, 2, 2});

  // modify data using .ptr() accessor

  float_t *ptr1 = tensor.host_ptr(0, 0, 0, 0);
  float_t *ptr2 = tensor.host_ptr(0, 1, 0, 0);

  for (size_t i = 0; i < 4; ++i) {
    ptr1[i] = float_t(1.0);
  }

  for (size_t i = 0; i < 4; ++i) {
    ptr2[i] = float_t(2.0);
  }

  // check data using operator[] accessor

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor.host_data()[i], float_t(1.0));
  }

  for (serial_size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor.host_data()[4 + i], float_t(2.0));
  }
}

TEST(tensor, access_data6) {
  Tensor<float_t, 4> tensor({1, 2, 2, 2});

  // modify data using .at() accessor

  tensor.host_at(0, 0, 0, 0) = float_t(1.0);
  tensor.host_at(0, 0, 1, 0) = float_t(1.0);
  tensor.host_at(0, 1, 0, 0) = float_t(1.0);
  tensor.host_at(0, 1, 1, 0) = float_t(1.0);

  tensor.host_at(0, 0, 0, 1) = float_t(2.0);
  tensor.host_at(0, 0, 1, 1) = float_t(2.0);
  tensor.host_at(0, 1, 0, 1) = float_t(2.0);
  tensor.host_at(0, 1, 1, 1) = float_t(2.0);

  // check data using .at() accessor

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(tensor.host_at(0, i, j, 0), float_t(1.0));
    }
  }

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(tensor.host_at(0, i, j, 1), float_t(2.0));
    }
  }
}

TEST(tensor, access_data7) {
  Tensor<float_t, 4> tensor({1, 2, 2, 2});

  // modify data using .at() accessor

  tensor.host_at(0, 0, 0, 0) = float_t(1.0);
  tensor.host_at(0, 0, 0, 1) = float_t(1.0);
  tensor.host_at(0, 0, 1, 0) = float_t(1.0);
  tensor.host_at(0, 0, 1, 1) = float_t(1.0);

  tensor.host_at(0, 1, 0, 0) = float_t(2.0);
  tensor.host_at(0, 1, 0, 1) = float_t(2.0);
  tensor.host_at(0, 1, 1, 0) = float_t(2.0);
  tensor.host_at(0, 1, 1, 1) = float_t(2.0);

  // check data using .ptr() accessor

  const float_t *ptr11 = tensor.host_ptr(0, 0, 0, 0);
  const float_t *ptr22 = tensor.host_ptr(0, 1, 0, 0);

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(ptr11[i], float_t(1.0));
  }

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(ptr22[i], float_t(2.0));
  }
}

TEST(tensor, access_data8) {
  Tensor<float_t, 4> tensor({1, 2, 2, 2});

  // modify data using .at() accessor

  tensor.host_at(0, 0, 0, 0) = float_t(1.0);
  tensor.host_at(0, 0, 0, 1) = float_t(1.0);
  tensor.host_at(0, 0, 1, 0) = float_t(1.0);
  tensor.host_at(0, 0, 1, 1) = float_t(1.0);

  tensor.host_at(0, 1, 0, 0) = float_t(2.0);
  tensor.host_at(0, 1, 0, 1) = float_t(2.0);
  tensor.host_at(0, 1, 1, 0) = float_t(2.0);
  tensor.host_at(0, 1, 1, 1) = float_t(2.0);

  // check data using operator[] accessor

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor.host_data()[i], float_t(1.0));
  }

  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor.host_data()[4 + i], float_t(2.0));
  }
}

/*TEST(tensor, access_data9) {
    Tensor<float_t> tensor({1,2,2,2});

    // modify data using operator[] accessor

    for (serial_size_t i = 0; i < 4; ++i) {
        tensor.mutable_host_data()[i] = float_t(1.0);
    }

    for (serial_size_t i = 0; i < 4; ++i) {
        tensor.mutable_host_data()[4 + i] = float_t(2.0);
    }

    // check data using operator[] accessor

    for (serial_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(tensor.host_data()[i], float_t(1.0));
    }

    for (serial_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(tensor.host_data()[4 + i], float_t(2.0));
    }
}*/

/*TEST(tensor, access_data10) {
    Tensor<float_t> tensor({1,2,2,2});

    // modify data using operator[] accessor

    for (serial_size_t i = 0; i < 4; ++i) {
        tensor.mutable_host_data()[i] = float_t(1.0);
    }

    for (serial_size_t i = 0; i < 4; ++i) {
        tensor.mutable_host_data()[4 + i] = float_t(2.0);
    }

    // check data using .at() accessor

    for (serial_size_t i = 0; i < 2; ++i) {
        for (serial_size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(tensor.host_at(0,0,i,j), float_t(1.0));
        }
    }

    for (serial_size_t i = 0; i < 2; ++i) {
        for (serial_size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(tensor.host_at(0,1,i,j), float_t(2.0));
        }
    }
}*/

/*TEST(tensor, access_data11) {
    Tensor<float_t> tensor({1,2,2,2});

    // modify data using operator[] accessor

    for (serial_size_t i = 0; i < 4; ++i) {
        tensor.mutable_host_data()[i] = float_t(1.0);
    }

    for (serial_size_t i = 0; i < 4; ++i) {
        tensor.mutable_host_data()[4 + i] = float_t(2.0);
    }

    // check data using .ptr() accessor

    const float_t* ptr11 = tensor.host_ptr(0,0,0,0);
    const float_t* ptr22 = tensor.host_ptr(0,1,0,0);

    for (serial_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ptr11[i], float_t(1.0));
    }

    for (serial_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ptr22[i], float_t(2.0));
    }
}*/

TEST(tensor, fill) {
  Tensor<float_t, 4> tensor({2, 2, 2, 2});

  // fill all tensor values with ones

  tensor.fill(float_t(1.0));

  for (size_t i = 0; i < tensor.size(); ++i) {
    EXPECT_EQ(tensor.host_data()[i], float_t(1.0));
  }

  // fill all tensor values with twos

  tensor.fill(float_t(2.0));

  for (size_t i = 0; i < tensor.size(); ++i) {
    EXPECT_EQ(tensor.host_data()[i], float_t(2.0));
  }
}

// TEST(tensor, linspace) {
//    Tensor<float_t> tensor(2,2,2,2);
//
//    // fill all tensor values with values from 1 to 16
//
//    tensor.linspace(float_t(1.0), float_t(16.0));
//
//    for (size_t i = 0; i < tensor.size(); ++i) {
//        EXPECT_EQ(tensor[i], float_t(1.0+i));
//    }
//
//    Tensor<float_t> tensor2(101,1,1,1);
//
//    // fill all tensor values with from 0 to 1
//
//    tensor2.linspace(float_t(0.0), float_t(1.0));
//
//    for (size_t i = 0; i < tensor2.size(); ++i) {
//        EXPECT_NEAR(tensor2[i], float_t(0.01*i), 1e-5);
//    }
//}

TEST(tensor, add1) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});
  Tensor<float_t, 4> t2({2, 2, 2, 2});

  // fill tensor with initial values

  t1.fill(float_t(1.0));
  t2.fill(float_t(3.0));

  // compute element-wise sum along all tensor values

  Tensor<float_t, 4> t3;

  layer_add(t3, t1, t2);

  // check that sum is okay

  for (size_t i = 0; i < t3.size(); ++i) {
    EXPECT_NEAR(t3.host_data()[i], float_t(4.0), 1e-5);
  }
}

TEST(tensor, add2a) {
  Tensor<float_t, 4> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(1.0));

  // compute element-wise sum along all tensor values

  Tensor<float_t, 4> t2;

  // check that sum is okay

  layer_add(t2, float_t(2.0), t);

  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_NEAR(t2.host_data()[i], float_t(3.0), 1e-5);
  }
}

TEST(tensor, add2b) {
  Tensor<float_t, 4> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(1.0));

  // compute element-wise sum along all tensor values

  Tensor<float_t, 4> t2;

  // check that sum is okay

  layer_add(t2, t, float_t(2.0));

  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_NEAR(t2.host_data()[i], float_t(3.0), 1e-5);
  }
}

TEST(tensor, add3) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});
  Tensor<float_t, 4> t2({4, 4, 4, 4});

  // compute element-wise sum along all tensor values.
  // Expect a throw since shapes are different

  Tensor<float_t, 4> t3;

  EXPECT_THROW(layer_add(t3, t1, t2);, nn_error);
}

TEST(tensor, sub1) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});
  Tensor<float_t, 4> t2({2, 2, 2, 2});

  // fill tensor with initial values

  t1.fill(float_t(1.0));
  t2.fill(float_t(3.0));

  // compute element-wise subtraction along all tensor values

  Tensor<float_t, 4> t3;
  layer_sub(t3, t1, t2);

  // check that sum is okay

  for (size_t i = 0; i < t3.size(); ++i) {
    EXPECT_NEAR(t3.host_data()[i], float_t(-2.0), 1e-5);
  }
}

TEST(tensor, sub2a) {
  Tensor<float_t, 4> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(1.0));

  // compute element-wise subtraction along all tensor values

  Tensor<float_t, 4> t2;

  layer_sub(t2, t, float_t(2.0));

  // check that subtraction is okay

  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_NEAR(t2.host_data()[i], float_t(-1.0), 1e-5);
  }
}

TEST(tensor, sub2b) {
  Tensor<float_t, 4> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(2.0));

  // compute element-wise subtraction along all tensor values

  Tensor<float_t, 4> t2;

  layer_sub(t2, float_t(1.0), t);

  // check that subtraction is okay

  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_NEAR(t2.host_data()[i], float_t(-1.0), 1e-5);
  }
}

TEST(tensor, sub3) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});
  Tensor<float_t, 4> t2({4, 4, 4, 4});

  // compute element-wise subtraction along all tensor values.
  // Expect a throw since shapes are different

  Tensor<float_t, 4> t3;

  EXPECT_THROW(layer_sub(t3, t1, t2), nn_error);
}

TEST(tensor, mul1) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});
  Tensor<float_t, 4> t2({2, 2, 2, 2});

  // fill tensor with initial values

  t1.fill(float_t(2.0));
  t2.fill(float_t(3.0));

  // compute element-wise multiplication along all tensor values

  Tensor<float_t, 4> t3;

  layer_mul(t3, t1, t2);

  // check that subtraction is okay

  for (size_t i = 0; i < t3.size(); ++i) {
    EXPECT_NEAR(t3.host_data()[i], float_t(6.0), 1e-5);
  }
}

TEST(tensor, mul2a) {
  Tensor<float_t, 4> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(2.0));

  // compute element-wise multiplication along all tensor values

  Tensor<float_t, 4> t2;

  layer_mul(t2, t, float_t(2.0));

  // check that multiplication is okay

  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_NEAR(t2.host_data()[i], float_t(4.0), 1e-5);
  }
}

TEST(tensor, mul2b) {
  Tensor<float_t, 4> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(2.0));

  // compute element-wise multiplication along all tensor values

  Tensor<float_t, 4> t2;

  layer_mul(t2, float_t(2.0), t);

  // check that multiplication is okay

  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_NEAR(t2.host_data()[i], float_t(4.0), 1e-5);
  }
}

TEST(tensor, mul3) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});
  Tensor<float_t, 4> t2({4, 4, 4, 4});

  // compute element-wise multiplication along all tensor values.
  // Expect a throw since shapes are different

  Tensor<float_t, 4> t3;

  EXPECT_THROW(layer_mul(t3, t1, t2), nn_error);
}

TEST(tensor, div1) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});
  Tensor<float_t, 4> t2({2, 2, 2, 2});

  // fill tensor with initial values

  t1.fill(float_t(1.0));
  t2.fill(float_t(2.0));

  // compute element-wise division along all tensor values

  Tensor<float_t, 4> t3;

  layer_div(t3, t1, t2);

  // check that division is okay

  for (size_t i = 0; i < t3.size(); ++i) {
    EXPECT_NEAR(t3.host_data()[i], float_t(0.5), 1e-5);
  }
}

TEST(tensor, div2a) {
  Tensor<float_t, 4> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(1.0));

  // compute element-wise division along all tensor values

  Tensor<float_t, 4> t2;

  layer_div(t2, t, float_t(2.0));

  // check that division is okay

  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_NEAR(t2.host_data()[i], float_t(0.5), 1e-5);
  }
}

TEST(tensor, div2b) {
  Tensor<float_t, 4> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(2.0));

  // compute element-wise division along all tensor values

  Tensor<float_t, 4> t2;

  layer_div(t2, float_t(1.0), t);

  // check that division is okay

  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_NEAR(t2.host_data()[i], float_t(0.5), 1e-5);
  }
}

TEST(tensor, div3) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});
  Tensor<float_t, 4> t2({4, 4, 4, 4});

  // compute element-wise division along all tensor values.
  // Expect a throw since shapes are different

  Tensor<float_t, 4> t3;

  EXPECT_THROW(layer_div(t3, t1, t2), nn_error);
}

TEST(tensor, div4) {
  Tensor<float_t, 4> t1({2, 2, 2, 2});
  Tensor<float_t, 4> t2({2, 2, 2, 2});

  // fill tensor with initial values

  t1.fill(float_t(1.0));
  t2.fill(float_t(0.0));

  // compute element-wise division along all tensor values

  Tensor<float_t, 4> t3;

  layer_div(t3, t1, t2);

  // check that division is NaN

  for (size_t i = 0; i < t3.size(); ++i) {
    EXPECT_TRUE(std::isnan(t3.host_data()[i]));
  }
}

TEST(tensor, div5) {
  Tensor<float_t, 4> t({2, 2, 2, 2});

  // fill tensor with initial values

  t.fill(float_t(1.0));

  // compute element-wise division along all tensor values

  Tensor<float_t, 4> t2;

  layer_div(t2, t, float_t(0.0));

  // check that division is NaN

  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_TRUE(std::isnan(t2.host_data()[i]));
  }
}

TEST(tensor, sqrt1) {
  Tensor<float_t, 4> t({2, 2, 2, 2});

  // fill tensor with initial values
  t.fill(float_t(4.0));

  // compute element-wise square root along all tensor values

  Tensor<float_t, 4> t2;

  layer_sqrt(t2, t);

  // check that root is okay

  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_NEAR(t2.host_data()[i], float_t(2.0), 1e-5);
  }
}

TEST(tensor, sqrt2) {
  Tensor<float_t, 4> t({2, 2, 2, 2});

  // fill tensor with initial values
  t.fill(float_t(-1.0));

  // compute element-wise square root along all tensor values

  Tensor<float_t, 4> t2;

  layer_sqrt(t2, t);

  // check that division is NaN

  for (size_t i = 0; i < t2.size(); ++i) {
    EXPECT_TRUE(std::isnan(t2.host_data()[i]));
  }
}

TEST(tensor, nd1) {
  Tensor<float_t, 3> t({3, 3, 3});

  // TODO(Randl): static assert tests
  /*EXPECT_THROW(t.host_at(0,1,1,0), nn_error);
  EXPECT_THROW(t.host_at(1,2), nn_error);*/
  EXPECT_THROW(t.host_at(4, 1, 1), nn_error);
  EXPECT_NO_THROW(t.host_at(2, 2, 2));

  for (size_t i = 0; i < 3; ++i) {
    t.host_at(i, i, i) = float_t(i * 0.5);
  }
}

template <size_t N>
std::ostream &print_tester(std::ostream &os) {
  os << "\nPrinting " << N << "-dimensional Tensor"
     << ":\n\n";
  std::vector<size_t> shape(N, 2);
  shape.back() = 3;

  Tensor<float_t, N> t(shape);
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

// TEST(tensor, exp) {
//    Tensor<float_t> t(2, 2, 2, 2);
//
//    // fill tensor with initial values
//    t.linspace(float_t(1.0), float_t(16.0));
//
//    // compute element-wise exponent along all tensor values
//
//    Tensor<float_t> t2 = t.exp();
//
//    // check that exponent is okay
//
//    for (size_t i = 0; i < t2.size(); ++i) {
//        EXPECT_NEAR(t2[i], float_t(exp(float_t(i+1))), 1e-5);
//    }
//}

}  // namespace tiny-dnn
