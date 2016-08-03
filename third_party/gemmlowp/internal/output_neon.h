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

// output_neon.h: optimized NEON specializations of the templates in output.h.

#ifndef GEMMLOWP_INTERNAL_OUTPUT_NEON_H_
#define GEMMLOWP_INTERNAL_OUTPUT_NEON_H_

#include "output.h"

#include <arm_neon.h>

namespace gemmlowp {

// Definitions of Fragment types wrapping NEON vector types.
typedef Fragment<int32x4_t, 4, 1, MapOrder::ColMajor> NEONFragmentInt32x4x1;
typedef Fragment<int32x4x4_t, 16, 1, MapOrder::ColMajor> NEONFragmentInt32x16x1;
typedef Fragment<uint8x8_t, 4, 1, MapOrder::ColMajor> NEONFragmentUint8x4x1;
typedef Fragment<uint8x16_t, 16, 1, MapOrder::ColMajor> NEONFragmentUint8x16x1;

// The code in unpack_neon.h will whenever possible process
// 16 entries at once (4 SIMD vectors of 4 entries each at once),
// to offer the compiler better optimization opportunities, reducing
// register dependencies. From the perspective of interfacing with the output
// pipeline, this takes the form of passing Fragment types wrapping int32x4x4_t
// data. In most cases, such data is handled simply by handling separately its
// 4 int32x4_t components. This partial specialization handles that for
// arbitrary output stages implementing a int32x4_t path. Only some output
// stages below will override this to use custom code to handle int32x4x4_t
// data all at once (see OutputStageSaturatingCastToUint8 below).
template <typename OutputStageType>
struct OutputStageEvalImpl<OutputStageType, NEONFragmentInt32x16x1> {
  typedef NEONFragmentInt32x16x1 InputType;
  typedef NEONFragmentInt32x16x1 OutputType;
  typedef OutputStageEvalImpl<OutputStageType, NEONFragmentInt32x4x1>
      ImplInt32x4;
  OutputStageEvalImpl(const OutputStageType& s) : impl_int32x4(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    OutputType output;

    for (int i = 0; i < 4; i++) {
      output.data.val[i] =
          impl_int32x4.Eval(input.data.val[i], row + 4 * i, col);
    }
    return output;
  }

  ImplInt32x4 impl_int32x4;
};

// Implementation of OutputStageQuantizeDownInt32ToUint8Scale for
// NEONFragmentInt32x4x1
template <>
struct OutputStageEvalImpl<OutputStageQuantizeDownInt32ToUint8Scale,
                           NEONFragmentInt32x4x1> {
  typedef NEONFragmentInt32x4x1 InputType;
  typedef NEONFragmentInt32x4x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8Scale OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int, int) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const std::int32_t result_mult_int = output_stage.result_mult_int;
    const std::int32_t result_offset = output_stage.result_offset;
    const std::int32_t preshift_offset =
        (result_shift < 1) ? 0 : (1 << (result_shift - 1));
    const int32x4_t a = vaddq_s32(input, vdupq_n_s32(result_offset));
    const int32x4_t b =
        vmlaq_n_s32(vdupq_n_s32(preshift_offset), a, result_mult_int);
    return vshlq_s32(b, vdupq_n_s32(-result_shift));
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageQuantizeDownInt32ToUint8ScalePC for
// NEONFragmentInt32x4x1
template <>
struct OutputStageEvalImpl<
    OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Col>,
    NEONFragmentInt32x4x1> {
  typedef NEONFragmentInt32x4x1 InputType;
  typedef NEONFragmentInt32x4x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Col>
      OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const std::int32_t preshift_offset =
        (result_shift < 1) ? 0 : (1 << (result_shift - 1));
    const int32x4_t result_mult_int =
        vld1q_s32(output_stage.result_mult_int.data(row));
    const int32x4_t result_offset =
        vld1q_s32(output_stage.result_offset.data(row));
    const int32x4_t a = vaddq_s32(input, result_offset);
    const int32x4_t b =
        vmlaq_s32(vdupq_n_s32(preshift_offset), a, result_mult_int);
    return vshlq_s32(b, vdupq_n_s32(-result_shift));
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageQuantizeDownInt32ToUint8ScalePC for
// NEONFragmentInt32x4x1
template <>
struct OutputStageEvalImpl<
    OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Row>,
    NEONFragmentInt32x4x1> {
  typedef NEONFragmentInt32x4x1 InputType;
  typedef NEONFragmentInt32x4x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Row>
      OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const std::int32_t preshift_offset =
        (result_shift < 1) ? 0 : (1 << (result_shift - 1));
    const int32x4_t result_mult_int =
        vld1q_s32(output_stage.result_mult_int.data(col));
    const int32x4_t result_offset =
        vld1q_s32(output_stage.result_offset.data(row));
    const int32x4_t a = vaddq_s32(input, result_offset);
    const int32x4_t b =
        vmlaq_s32(vdupq_n_s32(preshift_offset), a, result_mult_int);
    return vshlq_s32(b, vdupq_n_s32(-result_shift));
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageSaturatingCastToUint8 for NEONFragmentInt32x4x1
template <>
struct OutputStageEvalImpl<OutputStageSaturatingCastToUint8,
                           NEONFragmentInt32x4x1> {
  typedef NEONFragmentInt32x4x1 InputType;
  typedef NEONFragmentUint8x4x1 OutputType;
  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalImpl(const OutputStage&) {}

  OutputType Eval(InputType input, int, int) const {
    int16x8_t q16 = vcombine_s16(vqmovn_s32(input), vdup_n_s16(0));
    return vqmovun_s16(q16);
  }
};

// In the case of OutputStageSaturatingCastToUint8, the handling of
// NEONFragmentInt32x16x1 data can be made much more efficient by handling
// it all at once, instead of as 4 separate int32x4 values as in the above
// generic partial specialization. This also avoids the poor (50%) register
// utilization of FragmentUint8x4x1: by handling 16 scalar values at once,
// we are able to fill a uint8x16_t.
template <>
struct OutputStageEvalImpl<OutputStageSaturatingCastToUint8,
                           NEONFragmentInt32x16x1> {
  typedef NEONFragmentInt32x16x1 InputType;
  typedef NEONFragmentUint8x16x1 OutputType;
  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalImpl(const OutputStage&) {}

  OutputType Eval(InputType input, int, int) const {
    int16x8_t q16[2];
    for (int i = 0; i < 2; i++) {
      q16[i] = vcombine_s16(vqmovn_s32(input.data.val[2 * i]),
                            vqmovn_s32(input.data.val[2 * i + 1]));
    }
    return vcombine_u8(vqmovun_s16(q16[0]), vqmovun_s16(q16[1]));
  }
};

// Implementation of OutputStageBiasAddition for NEONFragmentInt32x4x1
template <typename VectorType>
struct OutputStageEvalImpl<OutputStageBiasAddition<VectorType>,
                           NEONFragmentInt32x4x1> {
  typedef NEONFragmentInt32x4x1 InputType;
  typedef NEONFragmentInt32x4x1 OutputType;
  typedef OutputStageBiasAddition<VectorType> OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    int32x4_t bias;
    if (VectorType::kShape == VectorShape::Row) {
      bias = vdupq_n_s32(output_stage.bias_vector(col));
    } else {
      bias = vld1q_s32(output_stage.bias_vector.data(row));
    }
    return vaddq_s32(input, bias);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageClamp for NEONFragmentInt32x4x1
template <>
struct OutputStageEvalImpl<OutputStageClamp, NEONFragmentInt32x4x1> {
  typedef NEONFragmentInt32x4x1 InputType;
  typedef NEONFragmentInt32x4x1 OutputType;
  typedef OutputStageClamp OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int, int) const {
    const int32x4_t min = vdupq_n_s32(output_stage.min);
    const int32x4_t max = vdupq_n_s32(output_stage.max);
    return vminq_s32(vmaxq_s32(input, min), max);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageTanh for NEONFragmentInt32x4x1
template <>
struct OutputStageEvalImpl<OutputStageTanh, NEONFragmentInt32x4x1>
    : OutputStageTanhEvalImpl<NEONFragmentInt32x4x1> {
  OutputStageEvalImpl(const OutputStageTanh& output_stage)
      : OutputStageTanhEvalImpl(output_stage) {}
};

// Specialization of StoreFinalOutput for NEONFragmentUint8x4x1.
// This is quite inefficient, but we have no choice: instructions storing 32bit
// at once also assume 32bit alignment. In practice, this slowness is not a
// problem because we use the x16 path for most values.
template <typename DstType>
inline void StoreFinalOutput(NEONFragmentUint8x4x1 value, DstType* dst, int row,
                             int col) {
  vst1_lane_u8(dst->data(row + 0, col), value, 0);
  vst1_lane_u8(dst->data(row + 1, col), value, 1);
  vst1_lane_u8(dst->data(row + 2, col), value, 2);
  vst1_lane_u8(dst->data(row + 3, col), value, 3);
}

// Specialization of StoreFinalOutput for NEONFragmentUint8x16x1.
template <typename DstType>
inline void StoreFinalOutput(NEONFragmentUint8x16x1 value, DstType* dst,
                             int row, int col) {
  vst1q_u8(dst->data(row, col), value);
}

// Specialization of StoreFinalOutput for NEONFragmentInt32x4x1, storing into a
// int32 destination.
template <typename DstType>
inline void StoreFinalOutput(NEONFragmentInt32x4x1 value, DstType* dst, int row,
                             int col) {
  vst1q_s32(dst->data(row, col), value);
}

// Specialization of StoreFinalOutput for NEONFragmentInt32x16x1, storing into
// a int32 destination.
template <typename DstType>
inline void StoreFinalOutput(NEONFragmentInt32x16x1 value, DstType* dst,
                             int row, int col) {
  for (int i = 0; i < 4; i++) {
    vst1q_s32(dst->data(row + 4 * i, col), value.data.val[i]);
  }
}

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_OUTPUT_NEON_H_
