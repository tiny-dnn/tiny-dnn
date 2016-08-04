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

// output.h: processing the 32-bit accumulators output by the unpack
// stage, obtaining the final result matrix entries and storing them into
// the destination matrix.

#ifndef GEMMLOWP_INTERNAL_OUTPUT_H_
#define GEMMLOWP_INTERNAL_OUTPUT_H_

#include <cmath>
#include <tuple>
#include <type_traits>

#include "../public/output_stages.h"
#include "fixedpoint.h"

namespace gemmlowp {

// A Fragment is a small fixed-size matrix typically stored in one or
// a few architecture-specific SIMD vectors. Besides plain old scalar types
// such as int32_t, Fragment types are what can be used as input/output data
// types for output pipeline stages.
//
// More details:
//
// In the generic scalar code in this file, we have only implemented
// evaluation of output stages for scalar inputs (e.g. plain int32_t values).
// Other files (e.g. output_neon.h) are to provide SIMD paths by implementing
// evaluation of output stages for SIMD vector types. However, this raises
// the question of how the different values ("lanes") in a SIMD vector
// correspond to different values in the whole matrices. For simple entry-wise
// output stages, this doesn't matter, but for other output stages depending
// on position within the whole matrix, this does matter. To solve this
// problem, rather than implementing evaluation of output stages for raw
// SIMD vector types, we wrap SIMD vector types in "fragment" structs that
// bring the additional structure of "shape" i.e. mapping SIMD lanes to
// matrix entries, and we specialize evaluation of output stage for such
// fragment types. The Fragment template struct here is how we generate
// all fragment structs. For example, in output_neon.h, it may be specialized
// with DataType=int32x4_t, Rows=4, Cols=1. MapOrder doesn't matter for
// vector shapes. While Fragment is only used for SIMD paths, we leave it
// here in this platform-generic file because this same template should
// cover the needs of any SIMD architectures.
template <typename tDataType, int tRows, int tCols, MapOrder tOrder>
struct Fragment {
  typedef tDataType DataType;
  static const int kRows = tRows;
  static const int kCols = tCols;
  static const MapOrder kOrder = tOrder;

  Fragment() {}
  Fragment(const DataType& d) : data(d) {}
  operator DataType() const { return data; }

  DataType data;
};

typedef Fragment<std::int32_t, 1, 1, MapOrder::ColMajor> FragmentInt32x1x1;
typedef Fragment<std::uint8_t, 1, 1, MapOrder::ColMajor> FragmentUint8x1x1;

// OutputStageEvalImpl is the template that we specialize to provide
// implementations of each output stage for each type of input data.
//
// Each specialization provides a OutputType typedef and an Eval function
// returning OutputType. The OutputType typically depends on the InputType.
//
// There are two dimensions in which input data types can vary:
//   1. Different output stages may expect different data types. The
//      only hard constraint is that the first stage accepts int32, as
//      the unpack stage produces int32 accumulators.
//   2. For a given scalar data type such as int32, there is still the
//      possibility of having SIMD vector types such as NEON int32x4_t,
//      typically wrapped as "fragment" types, see struct Fragment.
//      Thus, there can be several OutputStageEvalImpl
//      specializations for a single OutputStageType, for different
//      InputType's.
template <typename OutputStageType, typename InputType>
struct OutputStageEvalImpl {
  // This generic template body should never be hit.
  static_assert(
      std::is_same<InputType, void>::value,
      "Unimplemented: missing implementation of this output pipeline stage "
      "for this data type. This would happen if some architecture-specific "
      "SIMD back-end (output_$arch.h) were incomplete.");

  OutputStageEvalImpl(const OutputStageType&) {}
};

// Implementation of OutputStageQuantizeDownInt32ToUint8Scale for scalar data
template <>
struct OutputStageEvalImpl<OutputStageQuantizeDownInt32ToUint8Scale,
                           FragmentInt32x1x1> {
  typedef FragmentInt32x1x1 InputType;
  typedef FragmentInt32x1x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8Scale OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int, int) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const std::int32_t result_mult_int = output_stage.result_mult_int;
    const std::int32_t result_offset = output_stage.result_offset;
    const std::int32_t kRoundingTerm =
        (result_shift < 1) ? 0 : (1 << (result_shift - 1));
    return ((input + result_offset) * result_mult_int + kRoundingTerm) >>
           result_shift;
  }

  const OutputStage& output_stage;
};

template <>
struct OutputStageEvalImpl<
    OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Col>,
    FragmentInt32x1x1> {
  typedef FragmentInt32x1x1 InputType;
  typedef FragmentInt32x1x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Col>
      OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const std::int32_t result_mult_int = output_stage.result_mult_int(row);
    const std::int32_t result_offset = output_stage.result_offset(row);
    const std::int32_t kRoundingTerm =
        (result_shift < 1) ? 0 : (1 << (result_shift - 1));
    return ((input + result_offset) * result_mult_int + kRoundingTerm) >>
           result_shift;
  }

  const OutputStage& output_stage;
};

template <>
struct OutputStageEvalImpl<
    OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Row>,
    FragmentInt32x1x1> {
  typedef FragmentInt32x1x1 InputType;
  typedef FragmentInt32x1x1 OutputType;
  typedef OutputStageQuantizeDownInt32ToUint8ScalePC<VectorShape::Row>
      OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    const std::int32_t result_shift = output_stage.result_shift;
    const std::int32_t result_mult_int = output_stage.result_mult_int(col);
    const std::int32_t result_offset = output_stage.result_offset(col);
    const std::int32_t kRoundingTerm =
        (result_shift < 1) ? 0 : (1 << (result_shift - 1));
    return ((input + result_offset) * result_mult_int + kRoundingTerm) >>
           result_shift;
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageSaturatingCastToUint8 for scalar data
template <>
struct OutputStageEvalImpl<OutputStageSaturatingCastToUint8,
                           FragmentInt32x1x1> {
  typedef FragmentInt32x1x1 InputType;
  typedef FragmentUint8x1x1 OutputType;
  typedef OutputStageSaturatingCastToUint8 OutputStage;

  OutputStageEvalImpl(const OutputStage&) {}

  OutputType Eval(InputType input, int, int) const {
    std::int32_t data = input.data;
    return data > 255 ? 255 : data < 0 ? 0 : data;
  }
};

// Implementation of OutputStageBiasAddition for scalar data
template <typename VectorType>
struct OutputStageEvalImpl<OutputStageBiasAddition<VectorType>,
                           FragmentInt32x1x1> {
  typedef FragmentInt32x1x1 InputType;
  typedef FragmentInt32x1x1 OutputType;
  typedef OutputStageBiasAddition<VectorType> OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int row, int col) const {
    if (VectorType::kShape == VectorShape::Row) {
      return input + output_stage.bias_vector(col);
    } else {
      return input + output_stage.bias_vector(row);
    }
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageClamp for scalar data
template <>
struct OutputStageEvalImpl<OutputStageClamp, FragmentInt32x1x1> {
  typedef FragmentInt32x1x1 InputType;
  typedef FragmentInt32x1x1 OutputType;
  typedef OutputStageClamp OutputStage;

  OutputStageEvalImpl(const OutputStage& s) : output_stage(s) {}

  OutputType Eval(InputType input, int, int) const {
    const std::int32_t min = output_stage.min;
    const std::int32_t max = output_stage.max;
    return std::min(std::max(input.data, min), max);
  }

  const OutputStage& output_stage;
};

// Implementation of OutputStageTanh for either scalar or SIMD data
template <typename tInputType>
struct OutputStageTanhEvalImpl {
  typedef tInputType InputType;
  typedef InputType OutputType;
  typedef typename InputType::DataType DataType;
  typedef OutputStageTanh OutputStage;

  OutputStageTanhEvalImpl(const OutputStage& s) : output_stage(s) {
    const std::int32_t real_zero_as_int32 = output_stage.real_zero_as_int32;
    const std::int32_t real_amplitude_as_int32 =
        output_stage.real_amplitude_as_int32;

    input_cutoff_min = real_zero_as_int32 - 8 * real_amplitude_as_int32;
    input_cutoff_max = real_zero_as_int32 + 8 * real_amplitude_as_int32;
    output_min = real_zero_as_int32 - real_amplitude_as_int32;
    output_max = real_zero_as_int32 + real_amplitude_as_int32;

    double inverse_amplitude_normalized_double = 1.0 / real_amplitude_as_int32;
    inverse_amplitude_neg_exponent = 0;
    while (inverse_amplitude_normalized_double < 0.5) {
      inverse_amplitude_normalized_double *= 2;
      inverse_amplitude_neg_exponent++;
    }
    inverse_amplitude_normalized =
        ToFixedPoint<DataType, 0>(inverse_amplitude_normalized_double);

    double amplitude_normalized_double = real_amplitude_as_int32;
    amplitude_exponent = 0;
    while (amplitude_normalized_double >= 1.0) {
      amplitude_normalized_double *= 0.5;
      amplitude_exponent++;
    }
    amplitude_normalized =
        ToFixedPoint<DataType, 0>(amplitude_normalized_double);
  }

  OutputType Eval(InputType input, int, int) const {
    const std::int32_t real_zero_as_int32 = output_stage.real_zero_as_int32;

    typedef FixedPoint<DataType, 3> F3;
    typedef FixedPoint<DataType, 0> F0;

    // fixed-point affine transformation
    DataType input_centered =
        Sub(input.data, Dup<DataType>(real_zero_as_int32));
    F3 fixedpoint_input =
        F3::FromRaw(input_centered) * inverse_amplitude_normalized;
    // left shift
    fixedpoint_input.raw() =
        ShiftLeft(fixedpoint_input.raw(), 28 - inverse_amplitude_neg_exponent);
    // fixed-point tanh and multiplication
    F0 fixedpoint_output = tanh(fixedpoint_input) * amplitude_normalized;
    // right shift
    DataType int32_output =
        Add(Dup<DataType>(real_zero_as_int32),
            ShiftRight(fixedpoint_output.raw(), 31 - amplitude_exponent));

    DataType mask_if_below_cutoff_min =
        MaskIfLessThanOrEqual(input.data, Dup<DataType>(input_cutoff_min));
    DataType mask_if_above_cutoff_max =
        MaskIfGreaterThanOrEqual(input.data, Dup<DataType>(input_cutoff_max));

    return SelectUsingMask(
        mask_if_below_cutoff_min, Dup<DataType>(output_min),
        SelectUsingMask(mask_if_above_cutoff_max, Dup<DataType>(output_max),
                        int32_output));
  }

  const OutputStage& output_stage;
  std::int32_t input_cutoff_min, input_cutoff_max;
  std::int32_t output_min, output_max;
  FixedPoint<DataType, 0> inverse_amplitude_normalized;
  int inverse_amplitude_neg_exponent;
  FixedPoint<DataType, 0> amplitude_normalized;
  int amplitude_exponent;
};

template <>
struct OutputStageEvalImpl<OutputStageTanh, FragmentInt32x1x1>
    : OutputStageTanhEvalImpl<FragmentInt32x1x1> {
  OutputStageEvalImpl(const OutputStageTanh& output_stage)
      : OutputStageTanhEvalImpl(output_stage) {}
};

// OutputPipelineOutputType is a helper to determine the output data type of a
// pipeline, for a
// given input data type. It is a recursive template; see the explanation on
// OutputPipelineEvalImpl below.
template <typename OutputPipelineType, int FirstStage, typename InputType,
          bool StopRecursion =
              FirstStage == std::tuple_size<OutputPipelineType>::value>
struct OutputPipelineOutputType {
  typedef typename std::tuple_element<FirstStage, OutputPipelineType>::type
      FirstStageType;
  typedef typename OutputStageEvalImpl<FirstStageType, InputType>::OutputType
      FirstStageOutputType;
  typedef typename OutputPipelineOutputType<OutputPipelineType, FirstStage + 1,
                                            FirstStageOutputType>::Type Type;
};

template <typename OutputPipelineType, int FirstStage, typename InputType>
struct OutputPipelineOutputType<OutputPipelineType, FirstStage, InputType,
                                true> {
  typedef InputType Type;
};

// OutputPipelineEvalImpl is a helper to implement the evaluation of
// the whole pipeline. It is a recursive template to implement compile-time
// unrolling of the loop over all pipeline stages. The 'FirstStage' parameter
// is how we implement recursion: each specialization implements only
// evaluation starting at 'FirstStage'. The StopRecursion parameter is just a
// helper to implement the termination of the recursion as a partial
// specialization below.
template <typename OutputPipelineType, int FirstStage, typename InputType,
          bool StopRecursion =
              FirstStage == std::tuple_size<OutputPipelineType>::value>
struct OutputPipelineEvalImpl {
  typedef typename std::tuple_element<FirstStage, OutputPipelineType>::type
      FirstStageType;
  typedef typename OutputStageEvalImpl<FirstStageType, InputType>::OutputType
      FirstStageOutputType;
  typedef typename OutputPipelineOutputType<OutputPipelineType, FirstStage,
                                            InputType>::Type OutputType;

  OutputPipelineEvalImpl(const OutputPipelineType& output_pipeline)
      : head_impl(std::get<FirstStage>(output_pipeline)),
        tail_impl(output_pipeline) {}

  OutputType Eval(InputType input, int row, int col) const {
    // Evaluate the first stage.
    FirstStageOutputType first_stage_output = head_impl.Eval(input, row, col);
    // Recurse into the remaining stages.
    return tail_impl.Eval(first_stage_output, row, col);
  }

  const OutputStageEvalImpl<FirstStageType, InputType> head_impl;
  const OutputPipelineEvalImpl<OutputPipelineType, FirstStage + 1,
                               FirstStageOutputType>
      tail_impl;
};

// Specialization on 'StopRecursion' for terminating the recursion.
template <typename OutputPipelineType, int FirstStage, typename InputType>
struct OutputPipelineEvalImpl<OutputPipelineType, FirstStage, InputType, true> {
  OutputPipelineEvalImpl(const OutputPipelineType&) {}

  InputType Eval(InputType input, int, int) const {
    // Terminating the recursion.
    return input;
  }
};

// StoreFinalOutput takes the final value at the end of the output pipeline and
// stores it into the destination matrix. It can be specialized for different
// data types; the generic implementation here is typically used only for plain
// old scalar (not SIMD) types.
template <typename OutputType, typename DstType>
void StoreFinalOutput(OutputType value, DstType* dst, int row, int col) {
  *dst->data(row, col) = value;
}

template <typename OutputPipelineType, typename InputType>
struct OutputPipelineExecutor {
  OutputPipelineExecutor(const OutputPipelineType& output_pipeline)
      : output_pipeline_eval_impl_(output_pipeline) {}

  // RunOutputPipeline is the entry point into the output pipeline evaluation
  // code. It should be the only thing that unpack code calls. It takes the
  // result
  // of the unpack stage and stores it into the destination matrix.
  template <typename DstType>
  void Execute(InputType input, DstType* dst, int row, int col) {
    // Statically assert that the output pipeline matches the given destination
    // matrix's scalar type.
    typedef typename OutputPipelineOutputType<OutputPipelineType, 0,
                                              FragmentInt32x1x1>::Type::DataType
        ScalarOutputType;
    typedef typename DstType::Scalar ScalarDstType;
    static_assert(std::is_same<ScalarOutputType, ScalarDstType>::value,
                  "mismatched destination scalar type and output pipeline");

    // Evaluate the output pipeline.
    auto output = output_pipeline_eval_impl_.Eval(input, row, col);
    // Store the result into the destination matrix.
    StoreFinalOutput(output, dst, row, col);
  }

  const OutputPipelineEvalImpl<OutputPipelineType, 0, InputType>
      output_pipeline_eval_impl_;
};

}  // namespace gemmlowp

#endif  // GEMMLOWP_INTERNAL_OUTPUT_H_
