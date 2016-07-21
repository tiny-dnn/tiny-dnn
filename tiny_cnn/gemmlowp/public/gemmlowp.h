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

// gemmlowp.h: the main public interface header of gemmlowp.

#ifndef GEMMLOWP_PUBLIC_GEMMLOWP_H_
#define GEMMLOWP_PUBLIC_GEMMLOWP_H_
#include "../internal/kernel_default.h"
#include "../internal/multi_thread_gemm.h"
#include "../internal/unpack.h"
#include "bit_depth.h"
#include "map.h"
#include "output_stages.h"

namespace gemmlowp {

inline bool IsRequantizationWorthIt(int rows, int cols) {
  // We pack depth*(rows+cols) and compute depth*rows*cols.
  // Thus the ratio of compute/packing cost is rows*cols/(rows+cols)
  // In the square case rows==cols==N, it becomes N/2.
  return 2 * rows * cols >= (rows + cols) * kMinimumWidthForRequantization;
}

class GemmContext : public MultiThreadGemmContext {};

// Computes a general matrix product ("GEMM").
// This is a version that supports per channel quantization.
template <typename InputScalar, typename OutputScalar, typename BitDepthParams,
          MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder,
          typename LhsOffset, typename RhsOffset, typename OutputPipelineType>
void GemmWithOutputPipelinePC(GemmContext* context,
                              const MatrixMap<const InputScalar, LhsOrder>& lhs,
                              const MatrixMap<const InputScalar, RhsOrder>& rhs,
                              MatrixMap<OutputScalar, ResultOrder>* result,
                              const LhsOffset& lhs_offset,
                              const RhsOffset& rhs_offset,
                              const OutputPipelineType& output_pipeline) {
  assert(lhs.cols() == rhs.rows());

  int rows = result->rows();
  int cols = result->cols();
  int depth = lhs.cols();

  if (rows == 0 || cols == 0 || depth == 0) {
    // Vacuous GEMM, return early to avoid having to deal with
    // zero sizes below.
    return;
  }

  if (cols == 1) {
    if (IsRequantizationWorthIt(rows, cols)) {
      typedef DefaultKernel<KernelFamily::Gemv, BitDepthParams> Kernel;
      MultiThreadGemm<typename Kernel::Format, InputScalar, OutputScalar,
                      BitDepthParams>(context, Kernel(), lhs, rhs, result,
                                      lhs_offset, rhs_offset, output_pipeline);
    } else {
      typedef DefaultKernel<KernelFamily::Gemv, DefaultL8R8BitDepthParams>
          Kernel;
      MultiThreadGemm<typename Kernel::Format, InputScalar, OutputScalar,
                      DefaultL8R8BitDepthParams>(context, Kernel(), lhs, rhs,
                                                 result, lhs_offset, rhs_offset,
                                                 output_pipeline);
    }
  } else {
    if (IsRequantizationWorthIt(rows, cols)) {
      typedef DefaultKernel<KernelFamily::Gemm, BitDepthParams> Kernel;
      MultiThreadGemm<typename Kernel::Format, InputScalar, OutputScalar,
                      BitDepthParams>(context, Kernel(), lhs, rhs, result,
                                      lhs_offset, rhs_offset, output_pipeline);
    } else {
      typedef DefaultKernel<KernelFamily::Gemm, DefaultL8R8BitDepthParams>
          Kernel;
      MultiThreadGemm<typename Kernel::Format, InputScalar, OutputScalar,
                      DefaultL8R8BitDepthParams>(context, Kernel(), lhs, rhs,
                                                 result, lhs_offset, rhs_offset,
                                                 output_pipeline);
    }
  }
}

// Computes a general matrix product ("GEMM").
// This is the legacy version that does not support per channel quantization.
// The meaning of the offsets, result_mult_int and result_shift
// parameters is the same as in the standard EightBitIntGemm interface
// (which is also implemented in the eight_bit_int_gemm directory).
template <typename InputScalar, typename OutputScalar, typename BitDepthParams,
          MapOrder LhsOrder, MapOrder RhsOrder, MapOrder ResultOrder,
          typename OutputPipelineType>
void GemmWithOutputPipeline(GemmContext* context,
                            const MatrixMap<const InputScalar, LhsOrder>& lhs,
                            const MatrixMap<const InputScalar, RhsOrder>& rhs,
                            MatrixMap<OutputScalar, ResultOrder>* result,
                            int lhs_offset, int rhs_offset,
                            const OutputPipelineType& output_pipeline) {
  const OffsetColDup lhs_offset_vector(lhs_offset, lhs.rows());
  const OffsetRowDup rhs_offset_vector(rhs_offset, rhs.cols());
  GemmWithOutputPipelinePC<InputScalar, OutputScalar, BitDepthParams>(
      context, lhs, rhs, result, lhs_offset_vector, rhs_offset_vector,
      output_pipeline);
}

// Computes a general matrix product ("GEMM").
// The meaning of the offsets, result_mult_int and result_shift
// parameters is the same as in the standard EightBitIntGemm interface
// (which is also implemented in the eight_bit_int_gemm directory).
template <typename Scalar, typename BitDepthParams, MapOrder LhsOrder,
          MapOrder RhsOrder, MapOrder ResultOrder>
void Gemm(GemmContext* context, const MatrixMap<const Scalar, LhsOrder>& lhs,
          const MatrixMap<const Scalar, RhsOrder>& rhs,
          MatrixMap<Scalar, ResultOrder>* result, int lhs_offset,
          int rhs_offset, int result_offset, int result_mult_int,
          int result_shift) {
  GemmWithOutputPipeline<Scalar, Scalar, BitDepthParams>(
      context, lhs, rhs, result, lhs_offset, rhs_offset,
      MakeStandardOutputPipeline(result_offset, result_mult_int, result_shift));
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_GEMMLOWP_H_
