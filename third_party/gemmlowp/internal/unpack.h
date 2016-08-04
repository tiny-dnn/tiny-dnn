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

// unpack.h: unpacking the result blocks computed by compute.h,
// storing them into the destination matrix.

#ifndef GEMMLOWP_INTERNAL_UNPACK_H_
#define GEMMLOWP_INTERNAL_UNPACK_H_

#include "allocator.h"
#include "block_params.h"
#include "output.h"
#include "pack.h"

#include <cmath>

namespace gemmlowp {

class PackedResult {
 public:
  PackedResult(Allocator* _allocator, const BlockParams& _block_params)
      : allocator_(_allocator), block_params_(_block_params) {
    matrix_handle_ = allocator_->Reserve<std::int32_t>(block_params_.l2_rows *
                                                       block_params_.l2_cols);
  }

  ~PackedResult() {}

  MatrixMap<std::int32_t, MapOrder::ColMajor> Map() {
    return MatrixMap<std::int32_t, MapOrder::ColMajor>(
        allocator_->GetPointer<std::int32_t>(matrix_handle_),
        block_params_.l2_rows, block_params_.l2_cols, block_params_.l2_rows);
  }

  MatrixMap<const std::int32_t, MapOrder::ColMajor> Map() const {
    return MatrixMap<const std::int32_t, MapOrder::ColMajor>(
        allocator_->GetPointer<const std::int32_t>(matrix_handle_),
        block_params_.l2_rows, block_params_.l2_cols, block_params_.l2_rows);
  }

 private:
  Allocator* allocator_;
  Allocator::Handle matrix_handle_;
  const BlockParams& block_params_;
};

template <std::uint32_t numerator, std::uint32_t denominator>
std::int32_t RoundingMultiplyByConstantFraction(std::int32_t x) {
  if (numerator == denominator) {
    return x;
  }

  // We'll use only signed arithmetic here. This is
  // simpler (since this function operates on signed int32's) and
  // more friendly to ARM NEON, where this allows us to use the
  // VQRDMULH instruction.
  static const std::int32_t int_quotient =
      (numerator + denominator / 2) / denominator;
  static const std::int32_t remaining_numerator =
      numerator - int_quotient * denominator;
  static const std::int32_t scaled_remaining_numerator =
      static_cast<std::int32_t>(
          (static_cast<std::int64_t>(remaining_numerator) * (1ll << 31)) /
          denominator);

  const std::int64_t scaled_remaining_product =
      static_cast<std::int64_t>(x) *
      static_cast<std::int64_t>(scaled_remaining_numerator);

  const std::int32_t scaled_remaining_product_nudge =
      (scaled_remaining_product > 0 ? 1 : -1) * (1 << 30);

  const std::int32_t remaining_product = static_cast<std::int32_t>(
      (scaled_remaining_product + scaled_remaining_product_nudge) / (1u << 31));

  return x * int_quotient + remaining_product;
}

struct MatrixBlockBounds {
  int start_row;
  int start_col;
  int rows;
  int cols;

  MatrixBlockBounds(int start_row_, int start_col_, int rows_, int cols_)
      : start_row(start_row_), start_col(start_col_), rows(rows_), cols(cols_) {
  }
};

template <typename BitDepthParams, typename ResultBlockType,
          typename PackedResultType, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType>
struct UnpackResultImplGeneric {
  static void Unpack(ResultBlockType* dst, const MatrixBlockBounds& dst_block,
                     const PackedResultType& src, int depth,
                     const std::int32_t* lhs_sums_of_each_slice,
                     const std::int32_t* rhs_sums_of_each_slice,
                     const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                     const OutputPipelineType& output_pipeline) {
    assert(dst_block.start_row >= 0);
    assert(dst_block.start_row + dst_block.rows <= dst->rows());
    assert(dst_block.start_col >= 0);
    assert(dst_block.start_col + dst_block.cols <= dst->cols());
    auto src_map = src.Map();
    // No top-level blocking in the depth dimension at the moment.
    // Too much loss of precision.
    const int kLhsBits = BitDepthParams::LhsBitDepth::kBits;
    const int kRhsBits = BitDepthParams::RhsBitDepth::kBits;
    const std::int32_t kLhsMax = (1 << kLhsBits) - 1;
    const std::int32_t kRhsMax = (1 << kRhsBits) - 1;
    OutputPipelineExecutor<OutputPipelineType, FragmentInt32x1x1>
        output_pipeline_executor(output_pipeline);
    for (int c = 0; c < dst_block.cols; c++) {
      int c_dst = c + dst_block.start_col;
      for (int r = 0; r < dst_block.rows; r++) {
        int r_dst = r + dst_block.start_row;
        // To understand this code, read
        //   doc/low-precision.txt
        //   doc/less-than-8-bit.txt
        // We have 4 terms to sum: xx, x1, 1x, 11.
        // In case of requantization, we first need to scale them back
        // to the original scale, using RoundingMultiplyByConstantFraction.
        std::int32_t raw_xx = src_map(r, c);
        std::int32_t raw_x1 = lhs_sums_of_each_slice[r] * rhs_offset(c_dst);
        std::int32_t raw_1x = rhs_sums_of_each_slice[c] * lhs_offset(r_dst);
        std::int32_t term_xx =
            RoundingMultiplyByConstantFraction<255 * 255, kLhsMax * kRhsMax>(
                raw_xx);
        std::int32_t term_x1 =
            RoundingMultiplyByConstantFraction<255, kLhsMax>(raw_x1);
        std::int32_t term_1x =
            RoundingMultiplyByConstantFraction<255, kRhsMax>(raw_1x);
        std::int32_t term_11 = lhs_offset(r_dst) * rhs_offset(c_dst) * depth;
        // Sum the 4 terms.
        FragmentInt32x1x1 sum = term_xx + term_x1 + term_1x + term_11;

        output_pipeline_executor.Execute(sum, dst, r_dst, c_dst);
      }
    }
  }
};

template <typename BitDepthParams, typename ResultBlockType,
          typename PackedResultType, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType>
struct UnpackResultImpl
    : UnpackResultImplGeneric<BitDepthParams, ResultBlockType, PackedResultType,
                              LhsOffset, RhsOffset, OutputPipelineType> {};

template <typename BitDepthParams, typename ResultBlockType,
          typename PackedResultType, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType>
void UnpackResult(ResultBlockType* dst, const MatrixBlockBounds& dst_block,
                  const PackedResultType& src, int depth,
                  const std::int32_t* lhs_sums_of_each_slice,
                  const std::int32_t* rhs_sums_of_each_slice,
                  const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                  const OutputPipelineType& output_pipeline) {
  ScopedProfilingLabel label("unpack");
  UnpackResultImpl<BitDepthParams, ResultBlockType, PackedResultType,
                   LhsOffset, RhsOffset, OutputPipelineType>::Unpack(
      dst, dst_block, src, depth, lhs_sums_of_each_slice,
      rhs_sums_of_each_slice, lhs_offset, rhs_offset, output_pipeline);
}

}  // namespace gemmlowp

#ifdef GEMMLOWP_NEON
#include "unpack_neon.h"
#endif

#endif  // GEMMLOWP_INTERNAL_UNPACK_H_
