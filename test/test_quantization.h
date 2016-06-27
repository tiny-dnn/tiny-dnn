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
#include "picotest/picotest.h"
#include "testhelper.h"
#include "tiny_cnn/tiny_cnn.h"

namespace tiny_cnn {

TEST(QuantizationUtils, FloatToQuantized) {
  EXPECT_EQ(quint8(0), core::kernels::FloatToQuantized<quint8>(0.0f, 0.0f, 1.0f));
  EXPECT_EQ(quint8(0), core::kernels::FloatToQuantized<quint8>(0.0f, 0.0f, 2.0f));
  EXPECT_EQ(quint8(128), core::kernels::FloatToQuantized<quint8>(0.5f, 0.0f, 1.0f));
  EXPECT_EQ(quint8(128), core::kernels::FloatToQuantized<quint8>(1.0f, 0.0f, 2.0f));
  EXPECT_EQ(quint8(255), core::kernels::FloatToQuantized<quint8>(1.0f, 0.0f, 1.0f));
  EXPECT_EQ(quint8(255), core::kernels::FloatToQuantized<quint8>(2.0f, 0.0f, 2.0f));
  EXPECT_EQ(quint8(0), core::kernels::FloatToQuantized<quint8>(-128.0f, -128.0f, 127.0f));
  EXPECT_EQ(quint8(128), core::kernels::FloatToQuantized<quint8>(0.0f, -128.0f, 127.0f));
  EXPECT_EQ(quint8(255), core::kernels::FloatToQuantized<quint8>(127.0f, -128.0f, 127.0f));
  EXPECT_EQ(quint8(0), core::kernels::FloatToQuantized<quint8>(1.0f, 1.0f, 256.0f));
  EXPECT_EQ(quint8(127), core::kernels::FloatToQuantized<quint8>(128.0f, 1.0f, 256.0f));
  EXPECT_EQ(quint8(255), core::kernels::FloatToQuantized<quint8>(256.0f, 1.0f, 256.0f));

  const int int32_min = std::numeric_limits<int>::min();
  const int int32_max = std::numeric_limits<int>::max();

  EXPECT_EQ(qint32(int32_min),
            core::kernels::FloatToQuantized<qint32>(-128.0f, -128.0f, 128.0f));
  EXPECT_EQ(qint32(0), core::kernels::FloatToQuantized<qint32>(0.0f, -128.0f, 128.0f));
  EXPECT_EQ(qint32(int32_max),
            core::kernels::FloatToQuantized<qint32>(128.0f, -128.0f, 128.0f));
}

TEST(QuantizationUtils, QuantizedToFloat) {
  EXPECT_LT(fabsf(0.0f - core::kernels::QuantizedToFloat<quint8>(0, 0.0f, 1.0f)), 1 / 255.0f);
  EXPECT_LT(fabsf(0.0f - core::kernels::QuantizedToFloat<quint8>(0, 0.0f, 2.0f)), 1 / 255.0f);
  EXPECT_LT(fabsf(0.5f - core::kernels::QuantizedToFloat<quint8>(127, 0.0f, 1.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(1.0f - core::kernels::QuantizedToFloat<quint8>(127, 0.0f, 2.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(1.0f - core::kernels::QuantizedToFloat<quint8>(255, 0.0f, 1.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(2.0f - core::kernels::QuantizedToFloat<quint8>(255, 0.0f, 2.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(1.0f - core::kernels::QuantizedToFloat<quint8>(0, 1.0f, 256.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(128.0f - core::kernels::QuantizedToFloat<quint8>(127, 1.0f, 256.0f)),
            1 / 255.0f);
  EXPECT_LT(fabsf(256.0f - core::kernels::QuantizedToFloat<quint8>(255, 1.0f, 256.0f)),
            1 / 255.0f);

  const int int32_min = std::numeric_limits<int>::min();
  const int int32_max = std::numeric_limits<int>::max();

  EXPECT_LT(
      fabsf(-1.0f - core::kernels::QuantizedToFloat<qint32>(qint32(int32_min), -1.0f, 1.0f)),
      1e-5f);
  EXPECT_LT(fabsf(0.0f - core::kernels::QuantizedToFloat<qint32>(qint32(0), -1.0f, 1.0f)),
            1e-5f);
  EXPECT_LT(
      fabsf(1.0f - core::kernels::QuantizedToFloat<qint32>(qint32(int32_max), -1.0f, 1.0f)),
      1e-5f);
}

TEST(QuantizationUtils, AvoidBias) {
  for (int i = 0; i < 256; ++i) {
    const float as_float = core::kernels::QuantizedToFloat<quint8>(i, 0.0f, 2.0f);
    const int back_to_int = core::kernels::FloatToQuantized<quint8>(as_float, 0.0f, 2.0f);
    EXPECT_EQ(i, back_to_int);
  }
}

TEST(QuantizationUtils, RequantizeInNewRange) {
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
      const quint8 input_value =
          core::kernels::FloatToQuantized<quint8>(value_float, input_min, input_max);
      // Here we convert the quantized input value to what we expect
      // to get in the output range.
      const qint32 expected_value = core::kernels::FloatToQuantized<qint32>(
          core::kernels::QuantizedToFloat(input_value, input_min, input_max), output_min,
          output_max);
      EXPECT_EQ(expected_value,
                (core::kernels::RequantizeInNewRange<quint8, qint32>(
                    input_value, input_min, input_max, output_min, output_max)));
    }
  }
}

TEST(QuantizationUtils, RequantizeInNewRangeRealData) {
  const float value_as_float = -0.290169f;
  const float input_min = -0.739539f;
  const float input_max = 0.641057f;
  const float output_min = -2381.49f;
  const float output_max = 2207.6f;
  const quint8 value_as_quint8 =
      core::kernels::FloatToQuantized<quint8>(value_as_float, input_min, input_max);
  EXPECT_EQ(quint8(83), value_as_quint8);
  const qint32 actual_output = core::kernels::RequantizeInNewRange<quint8, qint32>(
      value_as_quint8, input_min, input_max, output_min, output_max);
  const qint32 value_as_qint32 =
      core::kernels::FloatToQuantized<qint32>(value_as_float, output_min, output_max);
  EXPECT_LT(std::abs(value_as_qint32 - actual_output), 10);
}

TEST(QuantizationUtils, RequantizeInNewRange32To8Bit) {
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
      const qint32 input_value =
          core::kernels::FloatToQuantized<qint32>(value_float, input_min, input_max);
      // Here we convert the quantized input value to what we expect
      // to get in the output range.
      const quint8 expected_value = core::kernels::FloatToQuantized<quint8>(
          core::kernels::QuantizedToFloat(input_value, input_min, input_max), output_min,
          output_max);
      EXPECT_EQ(expected_value,
                (core::kernels::RequantizeInNewRange<qint32, quint8>(
                    input_value, input_min, input_max, output_min, output_max)));
    }
  }
}

TEST(QuantizationUtils, RequantizeManyInNewRange32To8Bit) {
  // These are the float values we're going to test the conversions on.
  const size_t values_count = 6;
  const float values[values_count] = {0.0f, 0.45f, 1.0f, -1.0f, 127.0f, 255.0f};
  // These are the input and output ranges we'll test.
  const size_t ranges_count = 4;
  const float ranges[ranges_count][4] = {
      {0.0f, 255.0f, 0.0f, 255.0f},
      {0.0f, 1.0f, 0.0f, 1.0f},
      // should be better as below:
      // {-1.0f, 1.0f, -1.0f, 1.0f},
      // {-1.0f, 1.0f, -255.0f, 255.0f},
      {0.0f, 1.0f, 0.0f, 1.0f},
      {0.0f, 1.0f, 0.0f, 255.0f},
  };
  for (size_t range_index = 0; range_index < ranges_count; ++range_index) {
    const float input_min = ranges[range_index][0];
    const float input_max = ranges[range_index][1];
    const float output_min = ranges[range_index][2];
    const float output_max = ranges[range_index][3];
    qint32 values_quantized[values_count];
    quint8 expected_values[values_count];
    for (size_t value_index = 0; value_index < values_count; ++value_index) {
      const float value_float = values[value_index];
      values_quantized[value_index] =
          core::kernels::FloatToQuantized<qint32>(value_float, input_min, input_max);
      expected_values[value_index] = core::kernels::FloatToQuantized<quint8>(
          core::kernels::QuantizedToFloat(values_quantized[value_index], input_min, input_max),
          output_min, output_max);
    }
    quint8 output_values[values_count];
    core::kernels::RequantizeManyInNewRange<qint32, quint8>(values_quantized, values_count,
                                             input_min, input_max, output_min,
                                             output_max, output_values);
    for (size_t value_index = 0; value_index < values_count; ++value_index) {
      // Here we convert the quantized input value to what we expect
      // to get in the output range.
      EXPECT_EQ(expected_values[value_index], output_values[value_index]);
    }
  }
}

TEST(QuantizationUtils, FloatTensorToQuantized) {
  const int input_width = 3;
  const int input_height = 3;
  const float input_min = 0.0f;
  const float input_max = 255.0f;
  const vec_t input = {1.0f, -1.0f, 10.0f, 10.25f, 127.0f, 255.0f,
                                   512.0f, 0.0f, 23.0f};
  quint8 expected[9] = {1, 0, 10, 10, 127, 255, 255, 0, 23};
  std::vector<quint8> output = core::kernels::FloatTensorToQuantized<quint8>(input, input_min, input_max);
  for (size_t value_index = 0; value_index < 9; ++value_index) {
    EXPECT_EQ(expected[value_index], output[value_index]);
  }
}

TEST(QuantizationUtils, QuantizedTensorToFloat) {
  const int input_width = 3;
  const int input_height = 3;
  const float input_min = -128.0f;
  const float input_max = 127.0f;
  const vec_t input = {0, 128, 255, 23, 24, 25, 243, 244, 245};
  vec_t expected = {-128.0f, 0.0f, 127.0f, -105.0f, -104.0f,
                                      -103.0f, 115.0f, 116.0f, 117.0f};
  vec_t output = core::kernels::QuantizedTensorToFloat<quint8>(input, input_min, input_max);
  for (size_t value_index = 0; value_index < 9; ++value_index) {
    EXPECT_EQ(expected[value_index], output[value_index]);
  }
}

} // namespace tiny-cnn
