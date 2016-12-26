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

namespace tiny_dnn {
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
int64_t float_to_quantized_unclamped(float_t input,
                                     float_t range_min, float_t range_max) {
  if (range_min == range_max) {
    return 0;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (number_of_steps / range);
  int64_t quantized = static_cast<int64_t>(round(input * range_scale) -
                      round(range_min * range_scale));
  const int64_t lowest_quantized =
      static_cast<int64_t>(lowest<T>());
  quantized += lowest_quantized;
  return quantized;
}

inline int32_t int64_to_int32(int64_t src) {
    assert(src <= std::numeric_limits<int32_t>::max() &&
           src >= std::numeric_limits<int32_t>::min());
    return static_cast<int32_t>(src);
}

// This converts the float into the final quantized type, clamping/saturating
// any over or underflows.
template <class T>
T float_to_quantized(float_t input, float_t range_min, float_t range_max) {
  int64_t quantized = float_to_quantized_unclamped<T>(input,
                                                      range_min, range_max);
  const int64_t lowest_quantized =
      static_cast<int64_t>(lowest<T>());
  const int64_t highest_quantized =
      static_cast<int64_t>(highest<T>());
  quantized = std::max<int64_t>(quantized, lowest_quantized);
  quantized = std::min<int64_t>(quantized, highest_quantized);
  return static_cast<T>(static_cast<int32_t>(quantized));
}

template <class T>
float quantized_to_float(T input, float_t range_min, float_t range_max) {
  if (range_min == range_max) {
    return range_min;
  }
  const int number_of_bits = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust = (number_of_steps / (number_of_steps - 1.0));
  const double range = ((range_max - range_min) * range_adjust);
  const double range_scale = (range / number_of_steps);
  const int64_t lowest_quantized =
      static_cast<int64_t>(lowest<T>());
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  const double result = range_min + (offset_input * range_scale);
  return static_cast<float_t>(result);
}

template <class T>
float float_for_one_quantized_level(float_t range_min, float_t range_max) {
  const int64_t highest_ = static_cast<int64_t>(highest<T>());
  const int64_t lowest_ = static_cast<int64_t>(lowest<T>());
  const float float_for_one_quantized_level =
      (range_max - range_min) / (highest_ - lowest_);
  return float_for_one_quantized_level;
}

template <class T1, class T2, class T3>
void quantization_range_for_multiplication(float_t min_a, float_t max_a,
                                           float_t min_b, float_t max_b,
                                           float_t* min_c, float_t* max_c) {
  const float_t a_float_for_one_quant_level =
      float_for_one_quantized_level<T1>(min_a, max_a);
  const float_t b_float_for_one_quant_level =
      float_for_one_quantized_level<T2>(min_b, max_b);

  const int64_t c_highest = static_cast<int64_t>(highest<T3>());
  const int64_t c_lowest = static_cast<int64_t>(lowest<T3>());
  const float c_float_for_one_quant_level =
      a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

template <class T1, class T2>
inline T2 requantize_in_new_range(T1 input,
                                  float_t min_input, float_t max_input,
                                  float_t min_new, float_t max_new) {
  const float_t input_float =
      quantized_to_float<T1>(input, min_input, max_input);
  return float_to_quantized<T2>(input_float, min_new, max_new);
}

template <class T1, class T2>
inline void requantize_many_in_new_range(T1* input, size_t count,
                                         float_t min_input, float_t max_input,
                                         float_t min_output, float_t max_output,
                                         T2* output) {
  for (size_t index = 0; index < count; ++index) {
    const float_t input_float =
        quantized_to_float<T1>(input[index], min_input, max_input);
    output[index] = float_to_quantized<T2>(input_float, min_output, max_output);
  }
}

// Because converting 32-bit accumulated results down to eight bit is a common
// case, we have a specialized code path to handle it as efficiently as
// possible using only fixed-point math for the inner loop.
template <>
inline void requantize_many_in_new_range<int32_t, uint8_t>(
    int32_t* input, size_t count, float_t min_input, float_t max_input,
    float_t min_output, float_t max_output, uint8_t* output) {
  // Initially we calculate all the constants we need once, before we go into
  // the inner loop.
  const int fp_shift = 16;
  const float input_range = max_input - min_input;
  const float output_range = max_output - min_output;
  const float recip_output_range = (255.0f / output_range);
  const int64_t recip_output_range_fp =
      static_cast<int64_t>(recip_output_range * (1 << fp_shift));
  const int64_t range_scale_fp = static_cast<int64_t>(255.0f * (1 << fp_shift) *
      input_range / output_range);
  const int64_t input_offset_fp = static_cast<int64_t>(
      (min_input * recip_output_range_fp) + (range_scale_fp >> 1));
  const int64_t output_offset_fp = static_cast<int64_t>(
      round((min_output * 255.0f) / output_range));
  const int64_t rounding_delta = 1 << (fp_shift - 1);
  // Inside this loop we just do minimal adds, multiplies, and shifts, in a way
  // that could be easily adapted for a SIMD implementation. It should also be
  // possible to perform all the calculations in 32-bit rather than 64, but
  // that's not been implemented yet.
  for (size_t index = 0; index < count; ++index) {
    const int64_t input_value = static_cast<int64_t>(input[index]);
    const int64_t fp_value =
        ((input_value * range_scale_fp) >> 32) + input_offset_fp;
    const int64_t round_intermediate =
        ((fp_value >= 0) ? (fp_value + rounding_delta)
                         : (fp_value - rounding_delta)) >>
        fp_shift;
    int64_t quantized_int64 = (round_intermediate - output_offset_fp);
    quantized_int64 = std::max<int64_t>(quantized_int64, 0LL);
    quantized_int64 = std::min<int64_t>(quantized_int64, 255LL);
    output[index] = static_cast<uint8_t>(static_cast<int32_t>(quantized_int64));
  }
}

// REQUIRES: 'result->NumElements() == input.NumElements()'
template <class T>
void float_tensor_to_quantized_in_place(const vec_t& input,
                                        float_t min, float_t max,
                                        std::vector<T>* result) {
  const size_t data_size = input.size();
  for (size_t i = 0; i < data_size; ++i) {
    (*result)[i] = float_to_quantized<T>(input[i], min, max);
  }
}

template <class T>
std::vector<T> float_tensor_to_quantized(const vec_t& input,
                                         float_t min, float_t max) {
  std::vector<T> result(input.size(), static_cast<T>(0));
  float_tensor_to_quantized_in_place<T>(input, min, max, &result);
  return result;
}

// REQUIRES: 'result->NumElements() == input.NumElements()'
template <class T>
void quantized_tensor_to_float_in_place(const std::vector<T>& input,
                                        float_t min, float_t max,
                                        vec_t* result) {
  const size_t data_size = input.size();
  for (size_t i = 0; i < data_size; ++i) {
    (*result)[i] = quantized_to_float<T>(input[i], min, max);
  }
}

template <class T>
vec_t quantized_tensor_to_float(const std::vector<T>& input,
                                float_t min, float_t max) {
  vec_t result(input.size(), static_cast<float_t>(0));
  quantized_tensor_to_float_in_place<T>(input, min, max, &result);
  return result;
}

template <class T1, class T2>
void quantize_down_and_shrink_range(std::vector<T1>& input,
                                    float_t min_input, float_t max_input,
                                    float_t* min_new, float_t* max_new,
                                    std::vector<T2>* output){
  const int32_t input_lowest_quantized = static_cast<int32_t>(lowest<T1>());
  const int32_t input_highest_quantized = static_cast<int32_t>(highest<T1>());
  T1 actual_min_quantized = input_highest_quantized;
  T1 actual_max_quantized = input_lowest_quantized;
  for (serial_size_t i = 0; i < input.size(); ++i) {
    const T1 value = input[i];
    actual_min_quantized = std::min(actual_min_quantized, value);
    actual_max_quantized = std::max(actual_max_quantized, value);
  }
  // We want to make sure that the minimum is no larger than zero, so that the
  // convolution operation can run efficiently.
  *min_new = std::min(0.0f, quantized_to_float(actual_min_quantized, min_input,
                                      max_input));
  *max_new = quantized_to_float(actual_max_quantized, min_input, max_input);
  requantize_many_in_new_range<int32_t, uint8_t>(&input[0], input.size(),
                           min_input, max_input, *min_new,
                           *max_new, &(*output)[0]);
}

template <class T>
void get_output_min_and_max_for_quantized_add(float_t input_min, float_t input_max,
                                              float_t smaller_input_min,
                                              float_t smaller_input_max,
                                       float_t* output_min, float_t* output_max) {
  // We need to have a good range to add our two arguments together in. This
  // is surprisingly tricky, since it has to satisfy a few different needs:
  //  - Must be symmetrical around zero, so that 0 + 0 = 0.
  //  - Must hold the largest of the argument ranges.
  //  - Should have enough range that the bits of the lowest and highest
  //    arguments overlap if possible without the lower getting truncated.
  //  - Should have some headroom so that there's no overflow.
  //  - Needs to be signed.
  // This leads us to use a scheme where we (assuming the inputs are eight bit
  // and the output is 32-bit) use the bottom 32 - 17 = 15 bits to store the
  // accumulated results. This gives us all the properties we need.
  *output_max =
      std::max(input_max, std::max(-input_min, std::max(smaller_input_max,
                                                        -smaller_input_min))) *
      (1 << 17);
  *output_min = -(*output_max);
}

template <typename T1, typename T2, typename T3>
void quantized_add(const std::vector<T1>& input,
                  float_t input_min, float_t input_max,
                  const std::vector<T2>& smaller_input,
                  float_t smaller_input_min, float_t smaller_input_max,
                  std::vector<T3>* output,
                  float_t* output_min, float_t* output_max) {

  get_output_min_and_max_for_quantized_add<float_t>(input_min, input_max,
                                           smaller_input_min, smaller_input_max,
                                           output_min, output_max);
  // To do addition properly, we need to compensate for a possibly unbalanced
  // zero point in the total representation. The quantized value that
  // represents the real number zero needs to be subtracted before addition to
  // make sure that the identity of zero + zero = zero holds.
  const T3 zero_in_total_space =
      float_to_quantized<T3>(0.0f, *output_min, *output_max);

  const int64_t input_element_count = input.size();
  const int64_t smaller_input_element_count = smaller_input.size();

  float total_min = *output_min;
  float total_max = *output_max;
  const size_t how_many_iterations =
      (input_element_count / smaller_input_element_count);
  for (size_t iteration = 0; iteration < how_many_iterations; ++iteration) {
    const size_t offset = iteration * smaller_input_element_count;
    for (int c = 0; c < smaller_input_element_count; ++c) {
      const int index = (offset + c);
      // The two numbers we're going to add can each be in very different
      // ranges (e.g. the quantized value '127' may represent very different
      // real numbers in both) so we need to convert them to a common range
      // before we sum them.
      const T1 input_value = input[index];
      const T3 input_in_total_space = requantize_in_new_range<T1, T3>(
          input_value, input_min, input_max, total_min, total_max);
      const T2 smaller_input_value = smaller_input[c];
      const T3 smaller_input_in_total_space =
          requantize_in_new_range<T2, T3>(smaller_input_value, smaller_input_min,
                                       smaller_input_max, total_min, total_max);
      const T3 total_pre = input_in_total_space + smaller_input_in_total_space;
      // As noted above, we need to compensate for the offset of the actual
      // zero point in the space we're operating in.
      const T3 total = total_pre + zero_in_total_space;
      (*output)[index] = total;
    }
  }
}

// defined by Yida to simplify the API for quantization
template <class T>
vec_t tensor_range(const vec_t& input, float_t margin = 1e-3f) {
  vec_t result(2, static_cast<float_t>(input[0]));
  for (serial_size_t c = 0; c < input.size(); c++) {
      result[0] = std::min(result[0], input[c]);
      result[1] = std::max(result[1], input[c]);
  }
  if (result[0] == result[1]) {
    result[0] = input[0] - margin;
    result[1] = input[1] + margin;
  }
  return result;
}

template <class T>
void quantization_tensor(const vec_t& input, vec_t range,
    std::vector<T> quantized, int32_t offset) {
  range = tensor_range<float_t>(input);
  quantized = float_tensor_to_quantized<T>(input, range[0], range[1]);
  offset = float_to_quantized_unclamped<T>(0.0f, range[0], range[1]);
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
