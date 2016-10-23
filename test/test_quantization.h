/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.
    * Neither the name of the <organization> nor the
    names of its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once
 #include "gtest/gtest.h"
#include "testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

namespace tiny_dnn {

TEST(quantization_utils, float_to_quantized) {
  EXPECT_EQ(uint8_t(0), core::kernels::float_to_quantized<uint8_t>(0.0f, 0.0f, 1.0f));
  EXPECT_EQ(uint8_t(0), core::kernels::float_to_quantized<uint8_t>(0.0f, 0.0f, 2.0f));
  EXPECT_EQ(uint8_t(128), core::kernels::float_to_quantized<uint8_t>(0.5f, 0.0f, 1.0f));
  EXPECT_EQ(uint8_t(128), core::kernels::float_to_quantized<uint8_t>(1.0f, 0.0f, 2.0f));
  EXPECT_EQ(uint8_t(255), core::kernels::float_to_quantized<uint8_t>(1.0f, 0.0f, 1.0f));
  EXPECT_EQ(uint8_t(255), core::kernels::float_to_quantized<uint8_t>(2.0f, 0.0f, 2.0f));
  EXPECT_EQ(uint8_t(0), core::kernels::float_to_quantized<uint8_t>(-128.0f, -128.0f, 127.0f));
  EXPECT_EQ(uint8_t(128), core::kernels::float_to_quantized<uint8_t>(0.0f, -128.0f, 127.0f));
  EXPECT_EQ(uint8_t(255), core::kernels::float_to_quantized<uint8_t>(127.0f, -128.0f, 127.0f));
  EXPECT_EQ(uint8_t(0), core::kernels::float_to_quantized<uint8_t>(1.0f, 1.0f, 256.0f));
  EXPECT_EQ(uint8_t(127), core::kernels::float_to_quantized<uint8_t>(128.0f, 1.0f, 256.0f));
  EXPECT_EQ(uint8_t(255), core::kernels::float_to_quantized<uint8_t>(256.0f, 1.0f, 256.0f));

  const int int32_min = std::numeric_limits<int>::min();
  const int int32_max = std::numeric_limits<int>::max();

  EXPECT_EQ(int32_t(int32_min),
            core::kernels::float_to_quantized<int32_t>(-128.0f, -128.0f, 128.0f));
  EXPECT_EQ(int32_t(0), core::kernels::float_to_quantized<int32_t>(0.0f, -128.0f, 128.0f));
  EXPECT_EQ(int32_t(int32_max),
            core::kernels::float_to_quantized<int32_t>(128.0f, -128.0f, 128.0f));
}

TEST(quantization_utils, quantized_to_float) {
  EXPECT_LT(fabsf(0.0f - core::kernels::quantized_to_float<uint8_t>(0, 0.0f, 1.0f)), 1 / 255.0f);
  EXPECT_LT(fabsf(0.0f - core::kernels::quantized_to_float<uint8_t>(0, 0.0f, 2.0f)), 1 / 255.0f);
  EXPECT_LT(fabsf(127.0f/255.0f - core::kernels::quantized_to_float<uint8_t>(127, 0.0f, 1.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(2*127.0f/255.0f - core::kernels::quantized_to_float<uint8_t>(127, 0.0f, 2.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(1.0f - core::kernels::quantized_to_float<uint8_t>(255, 0.0f, 1.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(2.0f - core::kernels::quantized_to_float<uint8_t>(255, 0.0f, 2.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(1.0f - core::kernels::quantized_to_float<uint8_t>(0, 1.0f, 256.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(128.0f - core::kernels::quantized_to_float<uint8_t>(127, 1.0f, 256.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(256.0f - core::kernels::quantized_to_float<uint8_t>(255, 1.0f, 256.0f)),
            1 / 255.0f);

  const int int32_min = std::numeric_limits<int>::min();
  const int int32_max = std::numeric_limits<int>::max();

  EXPECT_LT(
      fabsf(-1.0f - core::kernels::quantized_to_float<int32_t>(int32_t(int32_min), -1.0f, 1.0f)),
      1e-5f);
  EXPECT_LT(fabsf(0.0f - core::kernels::quantized_to_float<int32_t>(int32_t(0), -1.0f, 1.0f)),
            1e-5f);
  EXPECT_LT(
      fabsf(1.0f - core::kernels::quantized_to_float<int32_t>(int32_t(int32_max), -1.0f, 1.0f)),
      1e-5f);
}

TEST(quantization_utils, avoid_bias) {
  for (int i = 0; i < 256; ++i) {
    const float as_float = core::kernels::quantized_to_float<uint8_t>(i, 0.0f, 2.0f);
    const int back_to_int = core::kernels::float_to_quantized<uint8_t>(as_float, 0.0f, 2.0f);
    EXPECT_EQ(i, back_to_int);
  }
}

TEST(quantization_utils, requantize_in_new_range) {
  // These are the float values we're going to test the conversions on.
  const size_t values_count = 6;
  const float values[values_count] = {0.0f, 0.5f, 1.0f, -1.0f, 127.0f, 255.0f};
  // These are the input and output ranges we'll test.
  const size_t ranges_count = 4;
  const float ranges[ranges_count][4] = {
      {0.0f, 255.0f, 0.0f, 255.0f},
      {0.0f, 1.0f, 0.0f, 1.0f},
      {-1.0f, 1.0f, -1.0f, 1.0f},
      {-1.0f, 1.0f, -255.0f, 255.0f},
  };
  for (size_t value_index = 0; value_index < values_count; ++value_index) {
    const float value_float = values[value_index];
    for (size_t range_index = 0; range_index < ranges_count; ++range_index) {
      const float input_min = ranges[range_index][0];
      const float input_max = ranges[range_index][1];
      const float output_min = ranges[range_index][2];
      const float output_max = ranges[range_index][3];
      const uint8_t input_value =
          core::kernels::float_to_quantized<uint8_t>(value_float, input_min, input_max);
      // Here we convert the quantized input value to what we expect
      // to get in the output range.
      const int32_t expected_value = core::kernels::float_to_quantized<int32_t>(
          core::kernels::quantized_to_float(input_value, input_min, input_max), output_min,
          output_max);
      EXPECT_EQ(expected_value,
                (core::kernels::requantize_in_new_range<uint8_t, int32_t>(
                    input_value, input_min, input_max, output_min, output_max)));
    }
  }
}

TEST(quantization_utils, requantize_in_new_range_real_data) {
  const float value_as_float = -0.290169f;
  const float input_min = -0.739539f;
  const float input_max = 0.641057f;
  const float output_min = -2381.49f;
  const float output_max = 2207.6f;
  const uint8_t value_as_uint8_t =
      core::kernels::float_to_quantized<uint8_t>(value_as_float, input_min, input_max);
  EXPECT_EQ(uint8_t(83), value_as_uint8_t);
  const int32_t actual_output = core::kernels::requantize_in_new_range<uint8_t, int32_t>(
      value_as_uint8_t, input_min, input_max, output_min, output_max);
  const int32_t value_as_int32_t =
      core::kernels::float_to_quantized<int32_t>(value_as_float, output_min, output_max);
  EXPECT_LT(std::abs(value_as_int32_t - actual_output), 10);
}

TEST(quantization_utils, requantize_in_new_range_32_to_8bit) {
  // These are the float values we're going to test the conversions on.
  const size_t values_count = 6;
  const float values[values_count] = {0.0f, 0.45f, 1.0f, -1.0f, 127.0f, 255.0f};
  // These are the input and output ranges we'll test.
  const size_t ranges_count = 4;
  const float ranges[ranges_count][4] = {
      {0.0f, 255.0f, 0.0f, 255.0f},
      {0.0f, 1.0f, 0.0f, 1.0f},
      {-1.0f, 1.0f, -1.0f, 1.0f},
      {-1.0f, 1.0f, -255.0f, 255.0f},
  };
  for (size_t value_index = 0; value_index < values_count; ++value_index) {
    const float value_float = values[value_index];
    for (size_t range_index = 0; range_index < ranges_count; ++range_index) {
      const float input_min = ranges[range_index][0];
      const float input_max = ranges[range_index][1];
      const float output_min = ranges[range_index][2];
      const float output_max = ranges[range_index][3];
      const int32_t input_value =
          core::kernels::float_to_quantized<int32_t>(value_float, input_min, input_max);
      // Here we convert the quantized input value to what we expect
      // to get in the output range.
      const uint8_t expected_value = core::kernels::float_to_quantized<uint8_t>(
          core::kernels::quantized_to_float(input_value, input_min, input_max), output_min,
          output_max);
      EXPECT_EQ(expected_value,
                (core::kernels::requantize_in_new_range<int32_t, uint8_t>(
                    input_value, input_min, input_max, output_min, output_max)));
    }
  }
}

TEST(quantization_utils, requantize_many_in_new_range_32_to_8bit) {
  // These are the float values we're going to test the conversions on.
  const size_t values_count = 6;
  const float values[values_count] = {0.0f, 0.45f, 1.0f, -1.0f, 127.0f, 255.0f};
  // These are the input and output ranges we'll test.
  const size_t ranges_count = 3;
  const float ranges[ranges_count][4] = {
      {0.0f, 255.0f, 0.0f, 255.0f},
      {0.0f, 1.0f, 0.0f, 1.0f},
      {-1.0f, 1.0f, -1.0f, 1.0f},
      // {-1.0f, 1.0f, -255.0f, 255.0f},
  };
  for (size_t range_index = 0; range_index < ranges_count; ++range_index) {
    const float input_min = ranges[range_index][0];
    const float input_max = ranges[range_index][1];
    const float output_min = ranges[range_index][2];
    const float output_max = ranges[range_index][3];
    int32_t values_quantized[values_count];
    uint8_t expected_values[values_count];
    for (size_t value_index = 0; value_index < values_count; ++value_index) {
      const float value_float = values[value_index];
      values_quantized[value_index] =
          core::kernels::float_to_quantized<int32_t>(value_float, input_min, input_max);
      expected_values[value_index] = core::kernels::float_to_quantized<uint8_t>(
          core::kernels::quantized_to_float(values_quantized[value_index], input_min, input_max),
          output_min, output_max);
    }
    uint8_t output_values[values_count];
    core::kernels::requantize_many_in_new_range<int32_t, uint8_t>(values_quantized, values_count,
                                             input_min, input_max, output_min,
                                             output_max, output_values);
    for (size_t value_index = 0; value_index < values_count; ++value_index) {
      // Here we convert the quantized input value to what we expect
      // to get in the output range.
      EXPECT_EQ(static_cast<float>(expected_values[value_index]), static_cast<float>(output_values[value_index]));
    }
  }
}

TEST(quantization_utils, float_tensor_to_quantized) {
  const float input_min = 0.0f;
  const float input_max = 255.0f;
  const vec_t input = {1.0f, -1.0f, 10.0f, 10.25f, 127.0f, 255.0f,
                                   512.0f, 0.0f, 23.0f};
  uint8_t expected[9] = {1, 0, 10, 10, 127, 255, 255, 0, 23};
  std::vector<uint8_t> output = core::kernels::float_tensor_to_quantized<uint8_t>(input, input_min, input_max);
  for (size_t value_index = 0; value_index < 9; ++value_index) {
    EXPECT_EQ(expected[value_index], output[value_index]);
  }
}

TEST(quantization_utils, quantized_tensor_to_float) {
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const std::vector<uint8_t> input = {0, 128, 255, 23, 24, 25, 243, 244, 245};
  vec_t expected = {-128.0f, 0.0f, 127.0f, -105.0f, -104.0f,
                                      -103.0f, 115.0f, 116.0f, 117.0f};
  vec_t output = core::kernels::quantized_tensor_to_float<uint8_t>(input, input_min, input_max);
  for (size_t value_index = 0; value_index < 9; ++value_index) {
    EXPECT_EQ(expected[value_index], output[value_index]);
  }
}

TEST(quantization_utils, quantize_down_and_shrink_range) {
  // For this test we have an input that has the theoretical range of -256.0f to
  // +256.0f, but the actual values present only span -1.0f to 1.0f. We expect
  // the operator to take advantage of this, and rescale the output to fill up
  // the available range in the lower bit depth, and update to the true min and
  // max ranges.
  const float_t input_min = -256.0f;
  const float_t input_max = 256.0f;
  float_t output_min;
  float_t output_max;
  std::vector<int32_t> input = {-(1 << 23), 0, (1 << 23)};
  std::vector<uint8_t> output(input.size(), static_cast<uint8_t>(0));
  core::kernels::quantize_down_and_shrink_range<int32_t, uint8_t>(input, input_min, input_max,
    &output_min, &output_max, &output);
  const std::vector<uint8_t> expected = {0, 127, 255};
  for (size_t value_index = 0; value_index < expected.size(); ++value_index) {
    EXPECT_EQ(expected[value_index], output[value_index]);
  }
  EXPECT_NEAR(-1.0f, output_min, 1E-5);
  EXPECT_NEAR(1.0f, output_max, 1E-5);
}

} // namespace tiny-dnn
