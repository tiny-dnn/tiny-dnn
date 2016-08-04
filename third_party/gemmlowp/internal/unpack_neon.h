// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// unpack_neon.h: optimized NEON specializations of the templates in unpack.h.

#ifndef GEMMLOWP_INTERNAL_UNPACK_NEON_H_
#define GEMMLOWP_INTERNAL_UNPACK_NEON_H_

#include "output_neon.h"
#include "unpack.h"

#include <arm_neon.h>

namespace gemmlowp {

template <std::uint32_t numerator, std::uint32_t denominator>
int32x4_t RoundingMultiplyByConstantFraction(int32x4_t x) {
  static_assert(numerator > 0 && denominator > 0,
                "only supporting positive num/denom");

  if (numerator == denominator) {
    return x;
  }

  static const std::int32_t int_quotient =
      (numerator + denominator / 2) / denominator;
  static const std::int32_t remaining_numerator =
      numerator - int_quotient * denominator;
  static const std::int32_t scaled_remaining_numerator =
      static_cast<std::int32_t>(
          (static_cast<std::int64_t>(remaining_numerator) * (1ll << 31)) /
          denominator);
  // Note: vqrdmulh instruction is rounding doubling multiply high.
  const int32x4_t remaining_product =
      vqrdmulhq_n_s32(x, scaled_remaining_numerator);

  return vmlaq_n_s32(remaining_product, x, int_quotient);
}

template <typename tScalar, VectorShape tShape>
int32x4_t get_int32x4_t_and_inc(
    ConstIterator<VectorMap<tScalar, tShape>>* iterator) {
  const int32x4_t result = vld1q_s32(iterator->get());
  *iterator += 4;
  return result;
}

template <typename tScalar, VectorShape tShape>
int32x4_t get_int32x4_t_and_inc(
    ConstIterator<VectorDup<tScalar, tShape>>* iterator) {
  const int32x4_t result = vdupq_n_s32(**iterator);
  // Increment really does nothing for VectorDup.
  *iterator += 4;
  return result;
}

template <typename BitDepthParams, typename PackedResultType,
          typename OutputScalar, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType>
struct UnpackResultImpl<BitDepthParams,
                        MatrixMap<OutputScalar, MapOrder::ColMajor>,
                        PackedResultType, LhsOffset, RhsOffset,
                        OutputPipelineType> {
  typedef MatrixMap<OutputScalar, MapOrder::ColMajor> ResultBlockType;
  static void Unpack(ResultBlockType* dst, const MatrixBlockBounds& dst_block,
                     const PackedResultType& src, int depth,
                     const std::int32_t* lhs_sums_of_each_slice,
                     const std::int32_t* rhs_sums_of_each_slice,
                     const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                     const OutputPipelineType& output_pipeline) {
    ScopedProfilingLabel label("optimized path (NEON)");
    assert(dst_block.start_row >= 0);
    assert(dst_block.start_row + dst_block.rows <= dst->rows());
    assert(dst_block.start_col >= 0);
    assert(dst_block.start_col + dst_block.cols <= dst->cols());
    const int kLhsBits = BitDepthParams::LhsBitDepth::kBits;
    const int kRhsBits = BitDepthParams::RhsBitDepth::kBits;
    const std::int32_t kLhsMax = (1 << kLhsBits) - 1;
    const std::int32_t kRhsMax = (1 << kRhsBits) - 1;
    auto src_map = src.Map();
    OutputPipelineExecutor<OutputPipelineType, FragmentInt32x1x1>
        output_pipeline_executor_int32x1x1(output_pipeline);
    OutputPipelineExecutor<OutputPipelineType, NEONFragmentInt32x4x1>
        output_pipeline_executor_int32x4x1(output_pipeline);
    OutputPipelineExecutor<OutputPipelineType, NEONFragmentInt32x16x1>
        output_pipeline_executor_int32x16x1(output_pipeline);

    for (int c = 0; c < dst_block.cols; c++) {
      int c_dst = c + dst_block.start_col;
      const std::int32_t* src_ptr = src_map.data(0, c);
      const std::int32_t* sums_of_each_slice_ptr = lhs_sums_of_each_slice;
      auto lhs_offset_iter = const_iterator(lhs_offset, dst_block.start_row);
      const std::int32_t rhs_offset_c = rhs_offset(c_dst);
      const std::int32_t rhs_sums_of_each_slice_c = rhs_sums_of_each_slice[c];

      // Handle 16 values at once for higher performance
      int dst_rows_aligned16 = RoundDown<16>(dst_block.rows);
      for (int r = 0; r < dst_rows_aligned16; r += 16) {
        int r_dst = r + dst_block.start_row;
        // Compute the sum of the 4 terms,
        //   q = term_xx + term_x1 + term_1x_plus_term_11
        // Refer to the generic code in unpack.h.
        int32x4_t raw_xx[4];
        for (int i = 0; i < 4; i++) {
          raw_xx[i] = vld1q_s32(src_ptr);
          src_ptr += 4;
        }
        int32x4_t raw_x1[4];
        for (int i = 0; i < 4; i++) {
          const int32x4_t sum_x1 = vld1q_s32(sums_of_each_slice_ptr);
          raw_x1[i] = vmulq_n_s32(sum_x1, rhs_offset_c);
          sums_of_each_slice_ptr += 4;
        }
        int32x4_t raw_1x[4];
        int32x4_t term_11[4];
        for (int i = 0; i < 4; i++) {
          const int32x4_t lhs_offsets = get_int32x4_t_and_inc(&lhs_offset_iter);
          raw_1x[i] = vmulq_n_s32(lhs_offsets, rhs_sums_of_each_slice_c);
          term_11[i] = vmulq_n_s32(lhs_offsets, rhs_offset_c * depth);
        }
        int32x4_t term_xx[4];
        for (int i = 0; i < 4; i++) {
          term_xx[i] =
              RoundingMultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(
                  raw_xx[i]);
        }
        int32x4_t term_x1[4];
        for (int i = 0; i < 4; i++) {
          term_x1[i] =
              RoundingMultiplyByConstantFraction<255, kLhsMax>(raw_x1[i]);
        }
        int32x4_t term_1x[4];
        for (int i = 0; i < 4; i++) {
          term_1x[i] =
              RoundingMultiplyByConstantFraction<255, kRhsMax>(raw_1x[i]);
        }
        int32x4x4_t q;
        for (int i = 0; i < 4; i++) {
          q.val[i] = vaddq_s32(vaddq_s32(term_xx[i], term_x1[i]),
                               vaddq_s32(term_1x[i], term_11[i]));
        }
        NEONFragmentInt32x16x1 f(q);
        output_pipeline_executor_int32x16x1.Execute(f, dst, r_dst, c_dst);
      }
      // We have finished handling groups of 16 entries at once; now
      // try to handle 4 entries at once.
      int dst_rows_aligned4 = RoundDown<4>(dst_block.rows);
      for (int r = dst_rows_aligned16; r < dst_rows_aligned4; r += 4) {
        int r_dst = r + dst_block.start_row;
        // Compute the sum of the 4 terms,
        //   q = term_xx + term_x1 + term_1x_plus_term_11
        // Refer to the generic code in unpack.h.
        const int32x4_t raw_xx = vld1q_s32(src_ptr);
        src_ptr += 4;
        const int32x4_t term_xx =
            RoundingMultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(
                raw_xx);
        const int32x4_t sum_x1 = vld1q_s32(sums_of_each_slice_ptr);
        const int32x4_t raw_x1 = vmulq_n_s32(sum_x1, rhs_offset_c);
        sums_of_each_slice_ptr += 4;
        const int32x4_t term_x1 =
            RoundingMultiplyByConstantFraction<255, kLhsMax>(raw_x1);
        const int32x4_t lhs_offsets = get_int32x4_t_and_inc(&lhs_offset_iter);
        const int32x4_t raw_1x =
            vmulq_n_s32(lhs_offsets, rhs_sums_of_each_slice_c);
        const int32x4_t term_1x =
            RoundingMultiplyByConstantFraction<255, kRhsMax>(raw_1x);
        const int32x4_t term_11 =
            vmulq_n_s32(lhs_offsets, rhs_offset_c * depth);
        int32x4_t q = vaddq_s32(vaddq_s32(term_xx, term_x1),
                                vaddq_s32(term_1x, term_11));
        NEONFragmentInt32x4x1 f(q);
        output_pipeline_executor_int32x4x1.Execute(f, dst, r_dst, c_dst);
      }
      // We have finished handling 4 entries at once; now handle
      // remaining entries one by one. This scalar code is similar
      // to the code in unpack.h, see comments there.
      for (int r = dst_rows_aligned4; r < dst_block.rows; r++) {
        int r_dst = r + dst_block.start_row;
        const std::int32_t raw_xx = src_map(r, c);
        const std::int32_t raw_x1 = lhs_sums_of_each_slice[r] * rhs_offset_c;
        const std::int32_t raw_1x =
            rhs_sums_of_each_slice_c * lhs_offset(r_dst);
        const std::int32_t term_xx =
            RoundingMultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(
                raw_xx);
        const std::int32_t term_x1 =
            RoundingMultiplyByConstantFraction<255, kLhsMax>(raw_x1);
        const std::int32_t term_1x =
            RoundingMultiplyByConstantFraction<255, kRhsMax>(raw_1x);
        const std::int32_t term_11 = lhs_offset(r) * rhs_offset(c) * depth;
        FragmentInt32x1x1 sum = term_xx + term_x1 + term_1x + term_11;
        output_pipeline_executor_int32x1x1.Execute(sum, dst, r_dst, c_dst);
      }
    }
  }
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_UNPACK_NEON_H_
