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

// output_stages.h: public definitions of the output stages that can
// be assembled into an output pipeline, to control how internal
// 32-bit accumulators are transformed to obtain the final uint8
// result matrix entries.

#ifndef GEMMLOWP_PUBLIC_OUTPUT_STAGES_H_
#define GEMMLOWP_PUBLIC_OUTPUT_STAGES_H_

#include <tuple>

#include "../internal/common.h"

namespace gemmlowp {

// This output stage takes int32 values and returns still int32 values,
// but "quantized down" to the uint8 scale; in other words, its output
// is typically what one would then clamp to [0..255] and cast to uint8
// (see OutputStageSaturatingCastToUint8).
//
// This "quantization down" process depends on 3 parameters,
//   result_offset, result_mult_int, result_shift,
// and the result is:
//   ((input + result_offset) * result_mult_int + rounding) >> result_shift
// where
//   rounding = (result_shift < 1) ? 0 : (1 << (result_shift - 1));
struct OutputStageQuantizeDownInt32ToUint8Scale {
  std::int32_t result_offset;
  std::int32_t result_mult_int;
  std::int32_t result_shift;
};

// This output stage takes int32 values and returns still int32 values,
// but "quantized down" to the uint8 scale; in other words, its output
// is typically what one would then clamp to [0..255] and cast to uint8
// (see OutputStageSaturatingCastToUint8).
//
// This "quantization down" process depends on 3 parameters,
//   result_offset, result_mult_int, result_shift,
// and the result is:
//   ((input + result_offset) * result_mult_int + rounding) >> result_shift
// where
//   rounding = (result_shift < 1) ? 0 : (1 << (result_shift - 1));
//
// Difference from OutputStageQuantizeDownInt32ToUint8Scale here is that each
// row or column of the output (depending on tShape) has its own result_offset
// and result_mult_int numbers.
template <VectorShape tShape>
struct OutputStageQuantizeDownInt32ToUint8ScalePC {
  VectorMap<const std::int32_t, tShape> result_offset;
  VectorMap<const std::int32_t, tShape> result_mult_int;
  std::int32_t result_shift;
};

// This output stage takes int32 values that are expected to be already
// on the final uint8 scale, but not necessarily in the [0..255] range.
// It clamps them to the [0..255] range and returns them casted to uint8.
struct OutputStageSaturatingCastToUint8 {};

// This output stage depends on a "bias vector" that should contain int32
// entries, and be either a row-vector of the same number of columns as the
// result matrix, or a column-vector of the same number of rows as the
// result matrix. This output stage takes int32 values and adds to them
// the corresponding entry of the bias vector (broadcasted in the other
// direction to fit the matrix's shape), outputting int32 values.
template <typename VectorType>
struct OutputStageBiasAddition {
  VectorType bias_vector;
};

// This output stage clamps value between the specified min and max bounds.
// It can be used to implement "rectified linear unit" activation functions
// in neural networks.
struct OutputStageClamp {
  std::int32_t min;
  std::int32_t max;
};

struct OutputStageTanh {
  std::int32_t real_zero_as_int32;
  std::int32_t real_amplitude_as_int32;
};

// An output pipeline is just a std::tuple of output stages.
// This function generates a standard output pipeline consisting of two stages:
// OutputStageQuantizeDownInt32ToUint8Scale, OutputStageSaturatingCastToUint8.
inline std::tuple<OutputStageQuantizeDownInt32ToUint8Scale,
                  OutputStageSaturatingCastToUint8>
MakeStandardOutputPipeline(std::int32_t result_offset,
                           std::int32_t result_mult_int,
                           std::int32_t result_shift) {
  OutputStageQuantizeDownInt32ToUint8Scale quantize_down_stage;
  quantize_down_stage.result_offset = result_offset;
  quantize_down_stage.result_mult_int = result_mult_int;
  quantize_down_stage.result_shift = result_shift;
  OutputStageSaturatingCastToUint8 saturating_cast_stage;
  return std::make_tuple(quantize_down_stage, saturating_cast_stage);
}

// An output pipeline is just a std::tuple of output stages.
// This function generates a standard output pipeline consisting of two stages:
// OutputStageQuantizeDownInt32ToUint8ScalePC, OutputStageSaturatingCastToUint8.
template <VectorShape tShape>
inline std::tuple<OutputStageQuantizeDownInt32ToUint8ScalePC<tShape>,
                  OutputStageSaturatingCastToUint8>
MakeStandardOutputPipeline(const VectorMap<const std::int32_t, tShape>&
                               result_offset,
                           const VectorMap<const std::int32_t, tShape>&
                               result_mult_int,
                           std::int32_t result_shift) {
  OutputStageQuantizeDownInt32ToUint8ScalePC<tShape> quantize_down_stage;
  quantize_down_stage.result_offset = result_offset;
  quantize_down_stage.result_mult_int = result_mult_int;
  quantize_down_stage.result_shift = result_shift;
  OutputStageSaturatingCastToUint8 saturating_cast_stage;
  return std::make_tuple(quantize_down_stage, saturating_cast_stage);
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_PUBLIC_OUTPUT_STAGES_H_
