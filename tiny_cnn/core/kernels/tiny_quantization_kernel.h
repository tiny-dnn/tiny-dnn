/*
    Copyright (c) 2016, Taiga Nomi, Edgar Riba
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

namespace tiny_cnn {
namespace core {
namespace kernels {

template <class T>
T highest() {
  return (std::numeric_limits<T>::max)();
}

template <class T>
T lowest()  {
  return std::numeric_limits<T>::is_integer ? (std::numeric_limits<T>::min)() : (-(std::numeric_limits<T>::max)());
}

// We have to be able to detect and handle overflows in int32, so this function
// uses doubles and int64's to make sure we have enough room.
template <class T>
int64 FloatToQuantizedUnclamped(float input, float range_min, float range_max) {
  if (range_min == range_max) {
    return 0;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64 number_of_steps = static_cast<int64>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (number_of_steps / range);
  int64 quantized =
      (round(input * range_scale) - round(range_min * range_scale));
  const int64 lowest_quantized =
      static_cast<double>(lowest<T>());
  quantized += lowest_quantized;
  return quantized;
}

// This converts the float into the final quantized type, clamping/saturating
// any over or underflows.
template <class T>
T FloatToQuantized(float input, float range_min, float range_max) {
  int64 quantized = FloatToQuantizedUnclamped<T>(input, range_min, range_max);
  const int64 lowest_quantized =
      static_cast<int64>(lowest<T>());
  const int64 highest_quantized =
      static_cast<int64>(highest<T>());
  quantized = std::max(quantized, lowest_quantized);
  quantized = std::min(quantized, highest_quantized);
  return static_cast<T>(static_cast<int32>(quantized));
}

template <class T>
float QuantizedToFloat(T input, float range_min, float range_max) {
  if (range_min == range_max) {
    return range_min;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64 number_of_steps = static_cast<int64>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (range / number_of_steps);
  const int64 lowest_quantized =
      static_cast<int64>(lowest<T>());
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  const double result = range_min + (offset_input * range_scale);
  return static_cast<float>(result);
}

template <class T>
float FloatForOneQuantizedLevel(float range_min, float range_max) {
  const int64 highest_ = static_cast<int64>(highest<T>());
  const int64 lowest_ = static_cast<int64>(lowest<T>());
  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest_ - lowest_);
  return float_for_one_quantized_level;
}

template <class T1, class T2, class T3>
void QuantizationRangeForMultiplication(float min_a, float max_a, float min_b,
                                        float max_b, float* min_c,
                                        float* max_c) {
  const float a_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T1>(min_a, max_a);
  const float b_float_for_one_quant_level =
      FloatForOneQuantizedLevel<T2>(min_b, max_b);

  const int64 c_highest = static_cast<int64>(highest<T3>());
  const int64 c_lowest = static_cast<int64>(lowest<T3>());
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

template <class T1, class T2>
inline T2 RequantizeInNewRange(T1 input, float min_input, float max_input,
                               float min_new, float max_new) {
  const float input_float = QuantizedToFloat<T1>(input, min_input, max_input);
  return FloatToQuantized<T2>(input_float, min_new, max_new);
}

template <class T1, class T2>
inline void RequantizeManyInNewRange(T1* input, size_t count, float min_input,
                                     float max_input, float min_output,
                                     float max_output, T2* output) {
  for (size_t index = 0; index < count; ++index) {
    const float input_float =
        QuantizedToFloat<T1>(input[index], min_input, max_input);
    output[index] = FloatToQuantized<T2>(input_float, min_output, max_output);
  }
}

// Because converting 32-bit accumulated results down to eight bit is a common
// case, we have a specialized code path to handle it as efficiently as
// possible using only fixed-point math for the inner loop.
template <>
inline void RequantizeManyInNewRange<qint32, quint8>(
    qint32* input, size_t count, float min_input, float max_input,
    float min_output, float max_output, quint8* output) {
  // Initially we calculate all the constants we need once, before we go into
  // the inner loop.
  const int fp_shift = 16;
  const float input_range = max_input - min_input;
  const float output_range = max_output - min_output;
  const float recip_output_range = (255.0 / output_range);
  const int64 recip_output_range_fp =
      static_cast<int64>(recip_output_range * (1 << fp_shift));
  const int64 range_scale_fp =
      static_cast<int64>(255.0 * (1 << fp_shift) * input_range / output_range);
  const int64 input_offset_fp =
      (min_input * recip_output_range_fp) + (range_scale_fp >> 1);
  const int64 output_offset_fp = round((min_output * 255.0) / output_range);
  const int64 rounding_delta = 1 << (fp_shift - 1);
  // Inside this loop we just do minimal adds, multiplies, and shifts, in a way
  // that could be easily adapted for a SIMD implementation. It should also be
  // possible to perform all the calculations in 32-bit rather than 64, but
  // that's not been implemented yet.
  for (size_t index = 0; index < count; ++index) {
    const int64 input_value = static_cast<int64>(input[index]);
    const int64 fp_value =
        ((input_value * range_scale_fp) >> 32) + input_offset_fp;
    const int64 round_intermediate =
        ((fp_value >= 0) ? (fp_value + rounding_delta)
                         : (fp_value - rounding_delta)) >>
        fp_shift;
    int64 quantized_int64 = (round_intermediate - output_offset_fp);
    quantized_int64 = std::max(quantized_int64, 0LL);
    quantized_int64 = std::min(quantized_int64, 255LL);
    output[index] = static_cast<quint8>(static_cast<int32>(quantized_int64));
  }
}

// REQUIRES: 'result->NumElements() == input.NumElements()'
template <class T>
void FloatTensorToQuantizedInPlace(const vec_t& input, float min, float max,
                                   std::vector<T>* result) {
  // DCHECK_EQ(DataTypeToEnum<T>::v(), result->dtype());
  // auto flat_input = input.flat<float>();
  // auto flat_result = result->flat<T>();
  const int data_size = input.size();
  // ASSERT_EQ(data_size, result->size());
  for (int i = 0; i < data_size; ++i) {
    (*result)[i] = FloatToQuantized<T>(input[i], min, max);
  }
}

template <class T>
std::vector<T> FloatTensorToQuantized(const vec_t& input, float min, float max) {
  std::vector<T> result(input.size(), static_cast<T>(0));
  FloatTensorToQuantizedInPlace<T>(input, min, max, &result);
  return result;
}

// REQUIRES: 'result->NumElements() == input.NumElements()'
template <class T>
void QuantizedTensorToFloatInPlace(const vec_t& input, float min, float max,
                                   vec_t* result) {
  // DCHECK_EQ(DataTypeToEnum<T>::v(), input.dtype());
  // auto flat_input = input.flat<T>();
  // auto flat_result = result->flat<float>();
  const int data_size = input.size();
  // ASSERT_EQ(data_size, result->size());
  for (int i = 0; i < data_size; ++i) {
    (*result)[i] = QuantizedToFloat<T>(input[i], min, max);
  }
}

template <class T>
vec_t QuantizedTensorToFloat(const vec_t& input, float min, float max) {
  vec_t result(input.size(), static_cast<float_t>(0));
  QuantizedTensorToFloatInPlace<T>(input, min, max, &result);
  return result;
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_cnn
