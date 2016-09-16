/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Implements a quantized eight-bit version of the matmul operation.

#include "third_party/gemmlowp/public/gemmlowp.h"
#include "tiny_dnn/core/kernels/tiny_quantization_kernel.h"

namespace tiny_dnn {
namespace core {
namespace kernels {

template <bool TransposeA, bool TransposeB, bool TransposeC>
void gemmlowp_multiply(const uint8_t*        a_data,
                      const uint8_t*        b_data,
                      int32_t*       c_data,
                      int m,
                      int n,
                      int k,
                      int offset_a,
                      int offset_b,
                      int lda,
                      int ldb,
                      int ldc) {
  const uint8_t* a_data_as_uint8 = a_data;
  const uint8_t* b_data_as_uint8 = b_data;
  int32_t* c_data_as_int32 = c_data;
  static const gemmlowp::MapOrder ResultOrder =
      !TransposeC ? gemmlowp::MapOrder::RowMajor : gemmlowp::MapOrder::ColMajor;
  static const gemmlowp::MapOrder LhsOrder =
      !TransposeA ? gemmlowp::MapOrder::RowMajor : gemmlowp::MapOrder::ColMajor;
  static const gemmlowp::MapOrder RhsOrder =
      !TransposeB ? gemmlowp::MapOrder::RowMajor : gemmlowp::MapOrder::ColMajor;
  gemmlowp::MatrixMap<const std::uint8_t, LhsOrder> lhs(a_data_as_uint8, m, k,
                                                        lda);
  gemmlowp::MatrixMap<const std::uint8_t, RhsOrder> rhs(b_data_as_uint8, k, n,
                                                        ldb);
  gemmlowp::MatrixMap<std::int32_t, ResultOrder> result(c_data_as_int32, m, n,
                                                        ldc);
  const std::tuple<> empty_pipeline = {};
  gemmlowp::GemmContext context;
  gemmlowp::GemmWithOutputPipeline<std::uint8_t, std::int32_t,
                                   gemmlowp::DefaultL8R8BitDepthParams>(
      &context, lhs, rhs, &result, -offset_a, -offset_b, empty_pipeline);
}

template <class T1, class T2, class Toutput>
void tiny_quantized_matmul(const std::vector<T1>&  a,
                          const std::vector<T2>&  b,
                          std::vector<Toutput>&   c,
                          const std::vector<size_t> shape_all,
                          const int32_t offset_a,
                          const int32_t offset_b,
                          const int32_t offset_c,
                          const int32_t mult_c,
                          const int32_t shift_c) {

    // Make sure that we have valid quantization ranges for the input buffers.
    // If the difference between the min and max is negative or zero, it makes
    // it hard to do meaningful intermediate operations on the values.

    int transpose_a_ = 0;
    int transpose_b_ = 1;
    int a_dim_remaining = 1 - transpose_a_;
    int b_dim_remaining = 1 - transpose_b_;

    const T1* a_data = &a[0];
    const T2* b_data = &b[0];
    Toutput* c_data = &c[0];

    const bool transpose_c = false;
    const size_t m = shape_all[a_dim_remaining];
    const size_t n = shape_all[2 + b_dim_remaining];
    const size_t k = shape_all[transpose_a_];
    const size_t lda = shape_all[1];
    const size_t ldb = shape_all[3];
    const size_t ldc = n;

    // The gemmlowp optimized library only works for a particular set of data
    // types, so check if we meet those requirements and
    // fall back to a slower reference implementation if not.
    if (std::is_same<T1, uint8_t>() && std::is_same<T2, uint8_t>() &&
        std::is_same<Toutput, int32_t>() && (offset_c == 0) && (mult_c == 1) &&
        (shift_c == 0) && (transpose_c == false)) {
      if (transpose_a_) {
        if (transpose_b_) {
          gemmlowp_multiply<true, true, false>(a_data, b_data, c_data, m, n, k,
                                              offset_a, offset_b, lda, ldb,
                                              ldc);
        } else {
          gemmlowp_multiply<true, false, false>(a_data, b_data, c_data, m, n, k,
                                               offset_a, offset_b, lda, ldb,
                                               ldc);
        }
      } else {
        if (transpose_b_) {
          gemmlowp_multiply<false, true, false>(a_data, b_data, c_data, m, n, k,
                                               offset_a, offset_b, lda, ldb,
                                               ldc);
        } else {
          gemmlowp_multiply<false, false, false>(a_data, b_data, c_data, m, n, k,
                                                offset_a, offset_b, lda, ldb,
                                                ldc);
        }
      }
    } /*else {
      ReferenceGemm<T1, T2, Toutput>(
          transpose_a_, transpose_b_, transpose_c, m, n, k, a_data, offset_a,
          lda, b_data, offset_b, ldb, c_data, shift_c, offset_c, mult_c, ldc);
    }

    float min_c_value;
    float max_c_value;
    quantization_range_for_multiplication<T1, T2, Toutput>(
        min_a, max_a, min_b, max_b, &min_c_value, &max_c_value);*/
  }

}
}
}  // namespace tiny_dnn
