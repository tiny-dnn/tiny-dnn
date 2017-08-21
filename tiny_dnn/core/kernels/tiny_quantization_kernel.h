/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <algorithm>
#include <limits>
#include <vector>

namespace tiny_dnn {
namespace core {
namespace kernels {

template <class T>
T highest() {
  return (std::numeric_limits<T>::max)();
}

template <class T>
T lowest() {
  return std::numeric_limits<T>::is_integer
           ? (std::numeric_limits<T>::min)()
           : (-(std::numeric_limits<T>::max)());
}

// We have to be able to detect and handle overflows in int32, so this function
// uses doubles and int64's to make sure we have enough room.
template <class T>
int64_t float_to_quantized_unclamped(float_t input,
                                     float_t range_min,
                                     float_t range_max) {
  if (range_min == range_max) {
    return 0;
  }
  const int number_of_bits      = sizeof(T) * 8;
  const int64_t number_of_steps = static_cast<int64_t>(1) << number_of_bits;
  const double range_adjust     = (number_of_steps / (number_of_steps - 1.0));
  const double range            = ((range_max - range_min) * range_adjust);
  const double range_scale      = (number_of_steps / range);
  int64_t quantized = static_cast<int64_t>(round(input * range_scale) -
                                           round(range_min * range_scale));
  const int64_t lowest_quantized = static_cast<int64_t>(lowest<T>());
  quantized += lowest_quantized;
  return quantized;
}

inline int32_t int64_to_int32(int64_t src) {
  assert(src <= std::numeric_limits<int32_t>::max() &&
         src >= std::numeric_limits<int32_t>::min());
  return static_cast<int32_t>(src);
}

/**
 * Converts the float into the final quantized type, clamping/saturating any
 * over or underflows.
 * @tparam T final quantized type
 * @param input
 * @param range_min
 * @param range_max
 * @return
 */
template <class T>
T float_to_quantized(float_t input, float_t range_min, float_t range_max) {
  int64_t quantized =
    float_to_quantized_unclamped<T>(input, range_min, range_max);
  const int64_t lowest_quantized  = static_cast<int64_t>(lowest<T>());
  const int64_t highest_quantized = static_cast<int64_t>(highest<T>());
  quantized = std::max<int64_t>(quantized, lowest_quantized);
  quantized = std::min<int64_t>(quantized, highest_quantized);
  return static_cast<T>(static_cast<int32_t>(quantized));
}

template <class T>
float quantized_to_float(T input, float_t range_min, float_t range_max) {
  if (range_min == range_max) {
    return range_min;
  }
  const int number_of_bits       = sizeof(T) * 8;
  const uint64_t number_of_steps = static_cast<uint64_t>(1) << number_of_bits;
  const double range_adjust      = (number_of_steps / (number_of_steps - 1.0));
  const double range             = ((range_max - range_min) * range_adjust);
  const double range_scale       = (range / number_of_steps);
  const int64_t lowest_quantized = static_cast<int64_t>(lowest<T>());
  const double offset_input = static_cast<double>(input) - lowest_quantized;
  const double result       = range_min + (offset_input * range_scale);
  return static_cast<float_t>(result);
}

template <class T>
float float_for_one_quantized_level(float_t range_min, float_t range_max) {
  const int64_t highest_ = static_cast<int64_t>(highest<T>());
  const int64_t lowest_  = static_cast<int64_t>(lowest<T>());
  const float float_for_one_quantized_level =
    (range_max - range_min) / (highest_ - lowest_);
  return float_for_one_quantized_level;
}

template <class T1, class T2, class T3>
void quantization_range_for_multiplication(float_t min_a,
                                           float_t max_a,
                                           float_t min_b,
                                           float_t max_b,
                                           float_t *min_c,
                                           float_t *max_c) {
  const float_t a_float_for_one_quant_level =
    float_for_one_quantized_level<T1>(min_a, max_a);
  const float_t b_float_for_one_quant_level =
    float_for_one_quantized_level<T2>(min_b, max_b);

  const int64_t c_highest = static_cast<int64_t>(highest<T3>());
  const int64_t c_lowest  = static_cast<int64_t>(lowest<T3>());
  const float c_float_for_one_quant_level =
    a_float_for_one_quant_level * b_float_for_one_quant_level;

  *min_c = c_float_for_one_quant_level * c_lowest;
  *max_c = c_float_for_one_quant_level * c_highest;
}

template <class T1, class T2>
inline T2 requantize_in_new_range(T1 input,
                                  float_t min_input,
                                  float_t max_input,
                                  float_t min_new,
                                  float_t max_new) {
  const float_t input_float =
    quantized_to_float<T1>(input, min_input, max_input);
  return float_to_quantized<T2>(input_float, min_new, max_new);
}

// TODO(Randl): refactor with iterators (stl-style)
template <class T1, class T2>
inline void requantize_many_in_new_range(T1 *input,
                                         size_t count,
                                         float_t min_input,
                                         float_t max_input,
                                         float_t min_output,
                                         float_t max_output,
                                         T2 *output) {
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
inline void requantize_many_in_new_range<int32_t, uint8_t>(int32_t *input,
                                                           size_t count,
                                                           float_t min_input,
                                                           float_t max_input,
                                                           float_t min_output,
                                                           float_t max_output,
                                                           uint8_t *output) {
  // Initially we calculate all the constants we need once, before we go into
  // the inner loop.
  const int fp_shift             = 16;
  const float input_range        = max_input - min_input;
  const float output_range       = max_output - min_output;
  const float recip_output_range = (255.0f / output_range);
  const int64_t recip_output_range_fp =
    static_cast<int64_t>(recip_output_range * (1 << fp_shift));
  const int64_t range_scale_fp =
    static_cast<int64_t>(255.0f * (1 << fp_shift) * input_range / output_range);
  const int64_t input_offset_fp = static_cast<int64_t>(
    (min_input * recip_output_range_fp) + (range_scale_fp >> 1));
  const int64_t output_offset_fp =
    static_cast<int64_t>(round((min_output * 255.0f) / output_range));
  const int64_t rounding_delta = 1 << (fp_shift - 1);
  // Inside this loop we just do minimal adds, multiplies, and shifts, in a
  // way
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
    quantized_int64         = std::max<int64_t>(quantized_int64, 0LL);
    quantized_int64         = std::min<int64_t>(quantized_int64, 255LL);
    output[index] = static_cast<uint8_t>(static_cast<int32_t>(quantized_int64));
  }
}

// REQUIRES: 'result->size() == input.size()'
template <typename Q, typename S>
void float_tensor_to_quantized_in_place(const Tensor<float_t, S> &input,
                                        float_t min,
                                        float_t max,
                                        Tensor<Q> *result) {
  auto in_iter  = input.host_begin();
  auto res_iter = result->host_begin();
  for (; in_iter != input.host_end(); ++in_iter, ++res_iter) {
    *res_iter = float_to_quantized<Q>(*in_iter, min, max);
  }
}

template <typename Q, typename S>
Tensor<Q> float_tensor_to_quantized(const Tensor<float_t, S> &input,
                                    float_t min,
                                    float_t max) {
  Tensor<Q> result(input.shape());
  result.fill(static_cast<Q>(0));
  float_tensor_to_quantized_in_place<Q>(input, min, max, &result);
  return result;
}

// REQUIRES: 'result->size() == input.size()'
template <typename Q, typename S>
void quantized_tensor_to_float_in_place(const Tensor<Q, S> &input,
                                        float_t min,
                                        float_t max,
                                        Tensor<> *result) {
  auto in_iter  = input.host_begin();
  auto res_iter = result->host_begin();
  for (; in_iter != input.host_end(); ++in_iter, ++res_iter) {
    *res_iter = quantized_to_float<Q>(*in_iter, min, max);
  }
}

template <typename Q, typename S>
Tensor<> quantized_tensor_to_float(const Tensor<Q, S> &input,
                                   float_t min,
                                   float_t max) {
  Tensor<> result({input.size()});
  result.fill(float_t(0));
  quantized_tensor_to_float_in_place<Q>(input, min, max, &result);
  return result;
}

template <class T1, class T2, class S1, class S2>
void quantize_down_and_shrink_range(Tensor<T1, S1> &input,
                                    float_t min_input,
                                    float_t max_input,
                                    float_t *min_new,
                                    float_t *max_new,
                                    Tensor<T2, S2> *output) {
  const T1 input_lowest_quantized  = lowest<T1>();
  const T1 input_highest_quantized = highest<T1>();

  auto in_minmax = std::minmax_element(input.host_begin(), input.host_end());
  T1 actual_min_quantized = std::min(input_highest_quantized, *in_minmax.first);
  T1 actual_max_quantized = std::max(input_lowest_quantized, *in_minmax.second);

  // We want to make sure that the minimum is no larger than zero, so that the
  // convolution operation can run efficiently.
  *min_new = std::min(
    0.0f, quantized_to_float(actual_min_quantized, min_input, max_input));
  *max_new = quantized_to_float(actual_max_quantized, min_input, max_input);
  requantize_many_in_new_range<T1, T2>(input.host_pbegin(), input.size(),
                                       min_input, max_input, *min_new, *max_new,
                                       output->host_pbegin());
}

/**
 * Quantize tensor to givin type, and return minimal and maximal elements by
 * reference
 * @tparam Q type to quantize
 * @tparam S
 * @param tensor Tensor to quantize
 * @param min_el minimal value returned by reference
 * @param max_el maximal value returned by reference
 * @return
 */
template <typename Q, typename S>
Tensor<Q> quantize_tensor(Tensor<float_t, S> tensor,
                          float_t &min_el,
                          float_t &max_el) {
  auto minmax = std::minmax_element(tensor.host_begin(), tensor.host_end());
  min_el      = *minmax.first;
  max_el      = *minmax.second;
  if (max_el == min_el) {
    // TODO(Randl): magic numbers
    min_el -= 10e-3;
    max_el += 10e-3;
  }
  return float_tensor_to_quantized<Q>(tensor, min_el, max_el);
}

}  // namespace kernels
}  // namespace core
}  // namespace tiny_dnn
