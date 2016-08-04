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

// single_thread_gemm.h: Single-threaded GEMM implementation.
// This is a good place to start reading code, as it shows the overall
// structure of a GEMM and is much simpler than multi_thread_gemm.h.

#ifndef GEMMLOWP_INTERNAL_SINGLE_THREAD_GEMM_H_
#define GEMMLOWP_INTERNAL_SINGLE_THREAD_GEMM_H_

#include <cassert>

#include "../public/map.h"
#include "allocator.h"
#include "compute.h"
#include "kernel.h"
#include "pack.h"
#include "unpack.h"

namespace gemmlowp {

class SingleThreadGemmContext {
 public:
  Allocator* allocator() { return &allocator_; }

 protected:
  Allocator allocator_;
};

typedef VectorMap<const int32_t, VectorShape::Col> OffsetColMap;
typedef VectorMap<const int32_t, VectorShape::Row> OffsetRowMap;
typedef VectorDup<const int32_t, VectorShape::Col> OffsetColDup;
typedef VectorDup<const int32_t, VectorShape::Row> OffsetRowDup;

template <typename KernelFormat, typename InputScalar, typename OutputScalar,
          typename BitDepthParams, MapOrder LhsOrder, MapOrder RhsOrder,
          MapOrder ResultOrder, typename LhsOffset, typename RhsOffset,
          typename OutputPipelineType>
void SingleThreadGemm(SingleThreadGemmContext* context,
                      const KernelBase& kernel,
                      const MatrixMap<const InputScalar, LhsOrder>& lhs,
                      const MatrixMap<const InputScalar, RhsOrder>& rhs,
                      MatrixMap<OutputScalar, ResultOrder>* result,
                      const LhsOffset& lhs_offset, const RhsOffset& rhs_offset,
                      const OutputPipelineType& output_pipeline) {
  ScopedProfilingLabel label("gemmlowp::SingleThreadGemm");

  assert(lhs.cols() == rhs.rows());

  int rows = result->rows();
  int cols = result->cols();
  int depth = lhs.cols();

  assert(rows > 0);
  assert(cols > 0);
  assert(depth > 0);

  Allocator* allocator = context->allocator();

  BlockParams block_params;
  block_params.Init<KernelFormat>(rows, cols, depth, 1);

  PackedSideBlock<typename KernelFormat::Lhs> packed_lhs(
      Side::Lhs, allocator, block_params);
  PackedSideBlock<typename KernelFormat::Rhs> packed_rhs(
      Side::Rhs, allocator, block_params);

  PackedResult packed_result(allocator, block_params);

  allocator->Commit();

  const bool pack_rhs_once = block_params.l2_cols == cols;

  if (pack_rhs_once) {
    PackRhs<BitDepthParams>(&packed_rhs, rhs);
  }

  for (int r = 0; r < rows; r += block_params.l2_rows) {
    int rs = std::min(block_params.l2_rows, rows - r);

    PackLhs<BitDepthParams>(&packed_lhs, lhs.block(r, 0, rs, depth));

    for (int c = 0; c < cols; c += block_params.l2_cols) {
      int cs = std::min(block_params.l2_cols, cols - c);

      if (!pack_rhs_once) {
        PackRhs<BitDepthParams>(&packed_rhs, rhs.block(0, c, depth, cs));
      }

      Compute(kernel, block_params, &packed_result, packed_lhs, packed_rhs);

      UnpackResult<BitDepthParams>(result, MatrixBlockBounds(r, c, rs, cs),
                                   packed_result, depth,
                                   packed_lhs.sums_of_each_slice(),
                                   packed_rhs.sums_of_each_slice(),
                                   lhs_offset, rhs_offset, output_pipeline);
    }
  }

  allocator->Decommit();
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_SINGLE_THREAD_GEMM_H_
