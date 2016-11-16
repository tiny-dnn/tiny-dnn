/*
    COPYRIGHT

    All contributions by Taiga Nomi
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    All other contributions:
    Copyright (c) 2013-2016, the respective contributors.
    All rights reserved.

    Each contributor holds copyright over their respective contributions.
    The project versioning (Git) records all such contribution source information.

    LICENSE

    The BSD 3-Clause License


    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.

    * Neither the name of tiny-dnn nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
#include "gtest/gtest.h"
#include "testhelper.h"

#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn;

namespace tiny_dnn {

TEST(tensor, shape) {
    Tensor<float_t> tensor(1,2,2,2);

    EXPECT_EQ(tensor.shape()[0], cnn_size_t(1));
    EXPECT_EQ(tensor.shape()[1], cnn_size_t(2));
    EXPECT_EQ(tensor.shape()[2], cnn_size_t(2));
    EXPECT_EQ(tensor.shape()[3], cnn_size_t(2));
}

TEST(tensor, size) {
    Tensor<float_t> tensor(2,2,2,2);

    EXPECT_EQ(tensor.size(), size_t(2*2*2*2));
}

TEST(tensor, check_bounds) {
    Tensor<float_t> tensor(1,2,2,1);

    // check bounds with .at() accessor

    EXPECT_NO_THROW(tensor.at<float_t>(0,0,0,0));
    EXPECT_NO_THROW(tensor.at<float_t>(0,0,1,0));
    EXPECT_NO_THROW(tensor.at<float_t>(0,1,0,0));
    EXPECT_NO_THROW(tensor.at<float_t>(0,1,1,0));

    EXPECT_THROW(tensor.at<float_t>(0,0,0,1), nn_error);
    EXPECT_THROW(tensor.at<float_t>(1,0,0,0), nn_error);
    EXPECT_THROW(tensor.at<float_t>(1,0,0,1), nn_error);

    // check bounds with .ptr() accessor

    EXPECT_NO_THROW(tensor.ptr<float_t>(0,0,0,0));
    EXPECT_NO_THROW(tensor.ptr<float_t>(0,0,1,0));
    EXPECT_NO_THROW(tensor.ptr<float_t>(0,1,0,0));
    EXPECT_NO_THROW(tensor.ptr<float_t>(0,1,1,0));

    EXPECT_THROW(tensor.ptr<float_t>(0,0,0,1), nn_error);
    EXPECT_THROW(tensor.ptr<float_t>(1,0,0,0), nn_error);
    EXPECT_THROW(tensor.ptr<float_t>(1,0,0,1), nn_error);

    // check bounds with operator[] accessor

    EXPECT_NO_THROW(tensor[0]);
    EXPECT_NO_THROW(tensor[3]);

    EXPECT_DEBUG_DEATH(tensor[4], "");
}

TEST(tensor, access_data1) {
    Tensor<float_t> tensor(1,2,2,1);

    const std::vector<cnn_size_t>& shape = tensor.shape();

    for (cnn_size_t n = 0; n < shape[0]; ++n) {
        for (cnn_size_t w = 0; w < shape[1]; ++w) {
            for (cnn_size_t h = 0; h < shape[2]; ++h) {
                for (cnn_size_t d = 0; d < shape[3]; ++d) {
                    EXPECT_EQ(tensor.at<float_t>(n,w,h,d),   float_t(0.0));
                    EXPECT_EQ(*tensor.ptr<float_t>(n,w,h,d), float_t(0.0));
                }
            }
        }
    }
}

TEST(tensor, access_data2) {
    Tensor<float_t> tensor(1,2,2,1);

    for (size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_EQ(tensor[i], float_t(0.0));
    }
}


TEST(tensor, access_data3) {
    Tensor<float_t> tensor(1,2,2,2);

    // modify data using .ptr() accessor

    float_t* ptr1 = tensor.ptr<float_t>(0,0,0,0);
    float_t* ptr2 = tensor.ptr<float_t>(0,0,0,1);

    for (cnn_size_t i = 0; i < 4; ++i) {
        ptr1[i] = float_t(1.0);
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        ptr2[i] = float_t(2.0);
    }

    // check data using .ptr() accessor

    const float_t* ptr11 = tensor.ptr<float_t>(0,0,0,0);
    const float_t* ptr22 = tensor.ptr<float_t>(0,0,0,1);

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ptr11[i], float_t(1.0));
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ptr22[i], float_t(2.0));
    }
}

TEST(tensor, access_data4) {
    Tensor<float_t> tensor(1,2,2,2);

    // modify data using .ptr() accessor

    float_t* ptr1 = tensor.ptr<float_t>(0,0,0,0);
    float_t* ptr2 = tensor.ptr<float_t>(0,0,0,1);

    for (cnn_size_t i = 0; i < 4; ++i) {
        ptr1[i] = float_t(1.0);
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        ptr2[i] = float_t(2.0);
    }

    // check data using .at() accessor

    for (cnn_size_t i = 0; i < 2; ++i) {
        for (cnn_size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(tensor.at<float_t>(0,i,j,0), float_t(1.0));
        }
    }

    for (cnn_size_t i = 0; i < 2; ++i) {
        for (cnn_size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(tensor.at<float_t>(0,i,j,1), float_t(2.0));
        }
    }
}

TEST(tensor, access_data5) {
    Tensor<float_t> tensor(1,2,2,2);

    // modify data using .ptr() accessor

    float_t* ptr1 = tensor.ptr<float_t>(0,0,0,0);
    float_t* ptr2 = tensor.ptr<float_t>(0,0,0,1);

    for (cnn_size_t i = 0; i < 4; ++i) {
        ptr1[i] = float_t(1.0);
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        ptr2[i] = float_t(2.0);
    }

    // check data using operator[] accessor

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(tensor[i], float_t(1.0));
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(tensor[4 + i], float_t(2.0));
    }
}

TEST(tensor, access_data6) {
    Tensor<float_t> tensor(1,2,2,2);

    // modify data using .at() accessor

    tensor.at<float_t>(0,0,0,0) = float_t(1.0);
    tensor.at<float_t>(0,0,1,0) = float_t(1.0);
    tensor.at<float_t>(0,1,0,0) = float_t(1.0);
    tensor.at<float_t>(0,1,1,0) = float_t(1.0);

    tensor.at<float_t>(0,0,0,1) = float_t(2.0);
    tensor.at<float_t>(0,0,1,1) = float_t(2.0);
    tensor.at<float_t>(0,1,0,1) = float_t(2.0);
    tensor.at<float_t>(0,1,1,1) = float_t(2.0);

    // check data using .at() accessor

    for (cnn_size_t i = 0; i < 2; ++i) {
        for (cnn_size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(tensor.at<float_t>(0,i,j,0), float_t(1.0));
        }
    }
    
    for (cnn_size_t i = 0; i < 2; ++i) {
        for (cnn_size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(tensor.at<float_t>(0,i,j,1), float_t(2.0));
        }
    }
}

TEST(tensor, access_data7) {
    Tensor<float_t> tensor(1,2,2,2);

    // modify data using .at() accessor

    tensor.at<float_t>(0,0,0,0) = float_t(1.0);
    tensor.at<float_t>(0,0,1,0) = float_t(1.0);
    tensor.at<float_t>(0,1,0,0) = float_t(1.0);
    tensor.at<float_t>(0,1,1,0) = float_t(1.0);

    tensor.at<float_t>(0,0,0,1) = float_t(2.0);
    tensor.at<float_t>(0,0,1,1) = float_t(2.0);
    tensor.at<float_t>(0,1,0,1) = float_t(2.0);
    tensor.at<float_t>(0,1,1,1) = float_t(2.0);

    // check data using .ptr() accessor

    const float_t* ptr11 = tensor.ptr<float_t>(0,0,0,0);
    const float_t* ptr22 = tensor.ptr<float_t>(0,0,0,1);

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ptr11[i], float_t(1.0));
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ptr22[i], float_t(2.0));
    }
}

TEST(tensor, access_data8) {
    Tensor<float_t> tensor(1,2,2,2);

    // modify data using .at() accessor

    tensor.at<float_t>(0,0,0,0) = float_t(1.0);
    tensor.at<float_t>(0,0,1,0) = float_t(1.0);
    tensor.at<float_t>(0,1,0,0) = float_t(1.0);
    tensor.at<float_t>(0,1,1,0) = float_t(1.0);

    tensor.at<float_t>(0,0,0,1) = float_t(2.0);
    tensor.at<float_t>(0,0,1,1) = float_t(2.0);
    tensor.at<float_t>(0,1,0,1) = float_t(2.0);
    tensor.at<float_t>(0,1,1,1) = float_t(2.0);

    // check data using operator[] accessor

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(tensor[i], float_t(1.0));
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(tensor[4 + i], float_t(2.0));
    }
}

TEST(tensor, access_data9) {
    Tensor<float_t> tensor(1,2,2,2);

    // modify data using operator[] accessor

    for (cnn_size_t i = 0; i < 4; ++i) {
        tensor[i] = float_t(1.0);
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        tensor[4 + i] = float_t(2.0);
    }

    // check data using operator[] accessor

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(tensor[i], float_t(1.0));
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(tensor[4 + i], float_t(2.0));
    }
}

TEST(tensor, access_data10) {
    Tensor<float_t> tensor(1,2,2,2);

    // modify data using operator[] accessor

    for (cnn_size_t i = 0; i < 4; ++i) {
        tensor[i] = float_t(1.0);
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        tensor[4 + i] = float_t(2.0);
    }

    // check data using .at() accessor

    for (cnn_size_t i = 0; i < 2; ++i) {
        for (cnn_size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(tensor.at<float_t>(0,i,j,0), float_t(1.0));
        }
    }

    for (cnn_size_t i = 0; i < 2; ++i) {
        for (cnn_size_t j = 0; j < 2; ++j) {
            EXPECT_EQ(tensor.at<float_t>(0,i,j,1), float_t(2.0));
        }
    }
}

TEST(tensor, access_data11) {
    Tensor<float_t> tensor(1,2,2,2);

    // modify data using operator[] accessor

    for (cnn_size_t i = 0; i < 4; ++i) {
        tensor[i] = float_t(1.0);
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        tensor[4 + i] = float_t(2.0);
    }

    // check data using .ptr() accessor

    const float_t* ptr11 = tensor.ptr<float_t>(0,0,0,0);
    const float_t* ptr22 = tensor.ptr<float_t>(0,0,0,1);

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ptr11[i], float_t(1.0));
    }

    for (cnn_size_t i = 0; i < 4; ++i) {
        EXPECT_EQ(ptr22[i], float_t(2.0));
    }
}

TEST(tensor, fill) {
    Tensor<float_t> tensor(2,2,2,2);

    // fill all tensor values with ones

    tensor.fill(float_t(1.0));

    for (size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_EQ(tensor[i], float_t(1.0));
    }
 
    // fill all tensor values with twos

    tensor.fill(float_t(2.0));

    for (size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_EQ(tensor[i], float_t(2.0));
    }
}

TEST(tensor, linspace) {
    Tensor<float_t> tensor(2,2,2,2);

    // fill all tensor values with values from 1 to 16

    tensor.linspace(float_t(1.0), float_t(16.0));

    for (size_t i = 0; i < tensor.size(); ++i) {
        EXPECT_EQ(tensor[i], float_t(1.0+i));
    }

    Tensor<float_t> tensor2(101,1,1,1);

    // fill all tensor values with from 0 to 1

    tensor2.linspace(float_t(0.0), float_t(1.0));

    for (size_t i = 0; i < tensor2.size(); ++i) {
        EXPECT_NEAR(tensor2[i], float_t(0.01*i), epsilon<float_t>());
    }
}

TEST(tensor, add1) {
    Tensor<float_t> t1(2,2,2,2);
    Tensor<float_t> t2(2,2,2,2);

    // fill tensor with initial values

    t1.fill(float_t(1.0));
    t2.fill(float_t(3.0));

    // compute element-wise sum along all tensor values

    Tensor<float_t> t3 = t1.add(t2);

    // check that sum is okay

    for (size_t i = 0; i < t3.size(); ++i) {
        EXPECT_NEAR(t3[i], float_t(4.0), epsilon<float_t>());
    }
}

TEST(tensor, add2) {
    Tensor<float_t> t(2,2,2,2);

    // fill tensor with initial values

    t.fill(float_t(1.0));

    // compute element-wise sum along all tensor values

    Tensor<float_t> t2 = t.add(float_t(2.0));

    // check that sum is okay

    for (size_t i = 0; i < t2.size(); ++i) {
        EXPECT_NEAR(t2[i], float_t(3.0), epsilon<float_t>());
    }
}

TEST(tensor, add3) {
    Tensor<float_t> t1(2,2,2,2);
    Tensor<float_t> t2(4,4,4,4);

    // compute element-wise sum along all tensor values.
    // Expect a throw since shapes are different

    EXPECT_THROW(t1.add(t2), nn_error);
}

TEST(tensor, sub1) {
    Tensor<float_t> t1(2,2,2,2);
    Tensor<float_t> t2(2,2,2,2);

    // fill tensor with initial values

    t1.fill(float_t(1.0));
    t2.fill(float_t(3.0));

    // compute element-wise subtraction along all tensor values

    Tensor<float_t> t3 = t1.sub(t2);

    // check that sum is okay

    for (size_t i = 0; i < t3.size(); ++i) {
        EXPECT_NEAR(t3[i], float_t(-2.0), epsilon<float_t>());
    }
}

TEST(tensor, sub2) {
    Tensor<float_t> t(2,2,2,2);

    // fill tensor with initial values

    t.fill(float_t(1.0));

    // compute element-wise subtraction along all tensor values

    Tensor<float_t> t2 = t.sub(float_t(2.0));

    // check that subtraction is okay

    for (size_t i = 0; i < t2.size(); ++i) {
        EXPECT_NEAR(t2[i], float_t(-1.0), epsilon<float_t>());
    }
}

TEST(tensor, sub3) {
    Tensor<float_t> t1(2,2,2,2);
    Tensor<float_t> t2(4,4,4,4);

    // compute element-wise subtraction along all tensor values.
    // Expect a throw since shapes are different

    EXPECT_THROW(t1.sub(t2), nn_error);
}

TEST(tensor, mul1) {
    Tensor<float_t> t1(2,2,2,2);
    Tensor<float_t> t2(2,2,2,2);

    // fill tensor with initial values

    t1.fill(float_t(2.0));
    t2.fill(float_t(3.0));

    // compute element-wise multiplication along all tensor values

    Tensor<float_t> t3 = t1.mul(t2);

    // check that subtraction is okay

    for (size_t i = 0; i < t3.size(); ++i) {
        EXPECT_NEAR(t3[i], float_t(6.0), epsilon<float_t>());
    }
}

TEST(tensor, mul2) {
    Tensor<float_t> t(2,2,2,2);

    // fill tensor with initial values

    t.fill(float_t(2.0));

    // compute element-wise multiplication along all tensor values

    Tensor<float_t> t2 = t.mul(float_t(2.0));

    // check that multiplication is okay

    for (size_t i = 0; i < t2.size(); ++i) {
        EXPECT_NEAR(t2[i], float_t(4.0), epsilon<float_t>());
    }
}

TEST(tensor, mul3) {
    Tensor<float_t> t1(2,2,2,2);
    Tensor<float_t> t2(4,4,4,4);

    // compute element-wise multiplication along all tensor values.
    // Expect a throw since shapes are different

    EXPECT_THROW(t1.mul(t2), nn_error);
}

TEST(tensor, div1) {
    Tensor<float_t> t1(2,2,2,2);
    Tensor<float_t> t2(2,2,2,2);

    // fill tensor with initial values

    t1.fill(float_t(1.0));
    t2.fill(float_t(2.0));

    // compute element-wise division along all tensor values

    Tensor<float_t> t3 = t1.div(t2);

    // check that division is okay

    for (size_t i = 0; i < t3.size(); ++i) {
        EXPECT_NEAR(t3[i], float_t(0.5), epsilon<float_t>());
    }
}

TEST(tensor, div2) {
    Tensor<float_t> t(2,2,2,2);

    // fill tensor with initial values

    t.fill(float_t(1.0));

    // compute element-wise division along all tensor values

    Tensor<float_t> t2 = t.div(float_t(2.0));

    // check that division is okay

    for (size_t i = 0; i < t2.size(); ++i) {
        EXPECT_NEAR(t2[i], float_t(0.5), epsilon<float_t>());
    }
}

TEST(tensor, div3) {
    Tensor<float_t> t1(2,2,2,2);
    Tensor<float_t> t2(4,4,4,4);

    // compute element-wise division along all tensor values.
    // Expect a throw since shapes are different

    EXPECT_THROW(t1.div(t2), nn_error);
}

TEST(tensor, div4) {
    Tensor<float_t> t1(2,2,2,2);
    Tensor<float_t> t2(2,2,2,2);

    // fill tensor with initial values

    t1.fill(float_t(1.0));
    t2.fill(float_t(0.0));

    // compute element-wise division along all tensor values

    Tensor<float_t> t3 = t1.div(t2);

    // check that division is NaN

    for (size_t i = 0; i < t3.size(); ++i) {
        EXPECT_TRUE(std::isnan(t3[i]));
    }
}

TEST(tensor, div5) {
    Tensor<float_t> t(2,2,2,2);

    // fill tensor with initial values

    t.fill(float_t(1.0));

    // compute element-wise division along all tensor values

    Tensor<float_t> t2 = t.div(float_t(0.0));

    // check that division is NaN

    for (size_t i = 0; i < t2.size(); ++i) {
        EXPECT_TRUE(std::isnan(t2[i]));
    }
}

TEST(tensor, sqrt1) {
    Tensor<float_t> t(2, 2, 2, 2);

    // fill tensor with initial values
    t.fill(float_t(4.0));

    // compute element-wise square root along all tensor values

    Tensor<float_t> t2 = t.sqrt();

    // check that root is okay

    for (size_t i = 0; i < t2.size(); ++i) {
        EXPECT_NEAR(t2[i], float_t(2.0), epsilon<float_t>());
    }
}

TEST(tensor, sqrt2) {
    Tensor<float_t> t(2, 2, 2, 2);

    // fill tensor with initial values
    t.fill(float_t(-1.0));

    // compute element-wise square root along all tensor values

    Tensor<float_t> t2 = t.sqrt();

    // check that division is NaN

    for (size_t i = 0; i < t2.size(); ++i) {
        EXPECT_TRUE(std::isnan(t2[i]));
    }
}

TEST(tensor, exp) {
    Tensor<float_t> t(2, 2, 2, 2);

    // fill tensor with initial values
    t.linspace(float_t(1.0), float_t(16.0));

    // compute element-wise exponent along all tensor values

    Tensor<float_t> t2 = t.exp();

    // check that exponent is okay

    for (size_t i = 0; i < t2.size(); ++i) {
        EXPECT_NEAR(t2[i], float_t(exp(float_t(i+1))), epsilon<float_t>());
    }
}

} // namespace tiny-dnn
